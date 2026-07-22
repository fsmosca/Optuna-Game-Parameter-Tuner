#!/usr/bin/env python


"""pymatch - built-in engine vs engine match manager.

Plays a match between two chess engines over the UCI or CECP/xboard
protocol using the python-chess library. It is the default match manager
of tuner.py (imported and called in-process via run_match) and can also
be run standalone:

    python pymatch.py --test-engine e1.exe --base-engine e2.exe \
        --openings book.epd --games 8 --depth 4 --concurrency 2

Standard chess only; use cutechess-cli for variants. The standalone CLI
prints cutechess-compatible score lines:

    Score of test vs base: W - L - D  [rate] games
    Finished match
"""

from __future__ import annotations

__version__ = 'v1.0.0'

import argparse
import datetime
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chess
import chess.engine
import chess.pgn


logger = logging.getLogger('pymatch')

# python-chess logs every engine I/O line at DEBUG; under the tuner the root
# logger writes DEBUG to log_tuner.txt, which would flood it with engine chat.
logging.getLogger('chess.engine').setLevel(logging.WARNING)

MATE_SCORE = 32000
# Safety cap: a game that somehow never ends is adjudicated as a draw.
MAX_PLIES = 1000


class MatchError(Exception):
    """Fatal match problem: bad config, or an engine that cannot (re)start."""


@dataclass
class EngineConfig:
    cmd: str                           # engine executable (path or PATH name)
    name: str                          # display name, e.g. 'test' / 'base'
    proto: str = 'uci'                 # 'uci' or 'xboard' ('cecp' is normalized)
    options: dict = field(default_factory=dict)
    cwd: Optional[str] = None          # default: the engine's own folder

    def __post_init__(self):
        if self.proto == 'cecp':
            self.proto = 'xboard'
        if self.proto not in ('uci', 'xboard'):
            raise MatchError(f'unsupported protocol "{self.proto}", use uci or cecp/xboard.')


@dataclass
class Adjudication:
    # Draw: after draw_movenumber full moves, when both engines report
    # |score| <= draw_score cp for draw_movecount consecutive moves each.
    draw_movenumber: Optional[int] = None    # None disables draw adjudication
    draw_movecount: int = 6
    draw_score: int = 0
    # Resign (two-sided): when one engine reports <= -resign_score and the
    # other >= +resign_score for resign_movecount consecutive moves each.
    resign_movecount: Optional[int] = None   # both must be set to enable
    resign_score: Optional[int] = None


@dataclass
class MatchConfig:
    test: EngineConfig
    base: EngineConfig
    games: int = 2                     # even; games//2 openings x 2 colors
    base_time_sec: float = 5.0
    inc_time_sec: float = 0.05
    depth: Optional[int] = None        # fixed depth per move
    nodes: Optional[int] = None        # fixed nodes per move
    opening_file: Optional[str] = None
    opening_format: str = 'epd'        # 'epd' (one fen/epd per line) or 'pgn'
    opening_order: str = 'random'      # 'random' or 'sequential'
    openings: Optional[List[str]] = None   # direct fen/epd list; overrides file
    adjudication: Adjudication = field(default_factory=Adjudication)
    pgnout: Optional[str] = None
    concurrency: int = 1
    event: str = 'pymatch'
    move_timeout: float = 300.0        # watchdog ceiling per move, depth/nodes mode
    time_margin: float = 5.0           # watchdog grace over the clock, tc mode
    max_restarts: int = 3              # engine crashes tolerated per engine slot


@dataclass
class GameRecord:
    round: int
    game_no: int                       # 1 or 2 within the round
    test_is_white: bool
    start_fen: str
    moves: list
    result: str                        # '1-0', '0-1' or '1/2-1/2'
    test_score: float                  # 1.0 / 0.5 / 0.0, test engine POV
    termination: str


@dataclass
class MatchResult:
    wins: int                          # test engine POV
    losses: int
    draws: int
    games: int
    records: List[GameRecord]

    @property
    def rate(self) -> float:
        return (self.wins + self.draws / 2) / self.games


def board_from_line(line: str) -> chess.Board:
    """Build a board from a full FEN or a 4-field EPD line."""
    line = line.strip()
    try:
        return chess.Board(line)
    except ValueError:
        board, _ = chess.Board.from_epd(line)
        return board


# Books are reloaded per match by the tuner loop; cache parsed books by
# path+mtime so a big book is not re-parsed on every trial.
_opening_cache = {}
_warned = set()
_warn_lock = threading.Lock()


def _warn_once(msg):
    with _warn_lock:
        if msg not in _warned:
            _warned.add(msg)
            logger.warning(msg)


def load_openings(path: str, fmt: str) -> List[str]:
    """
    Load a book as a list of start positions. epd = the raw fen/epd lines,
    kept unparsed: books run to hundreds of thousands of positions and a match
    plays only games/2 of them, so the lines are validated in select_openings()
    once picked (parsing a whole book costs minutes and hundreds of MB).
    pgn = the final position of each game's mainline, which must be parsed.
    """
    p = Path(path)
    if not p.is_file():
        raise MatchError(f'opening file not found: {path}')

    key = (str(p.resolve()), p.stat().st_mtime, fmt)
    if key in _opening_cache:
        return _opening_cache[key]

    openings = []
    if fmt == 'pgn':
        with open(p, encoding='utf-8-sig', errors='replace') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                openings.append(game.end().board().fen())
    else:
        with open(p, encoding='utf-8-sig') as f:
            openings = [line.strip() for line in f if line.strip()]
        # Fail early and clearly on a book that is not positions at all,
        # rather than at the first match.
        try:
            board_from_line(openings[0])
        except (ValueError, IndexError) as err:
            raise MatchError(f'first opening in {path} is not a valid fen/epd: {err}')

    if not openings:
        raise MatchError(f'no openings found in {path}')

    _opening_cache[key] = openings
    return openings


def mover_score(play_result) -> Optional[int]:
    """The cp score the engine reported for the move it just made, from its
    own point of view; None when the engine did not report a score."""
    score = play_result.info.get('score')
    if score is None:
        return None
    return score.relative.score(mate_score=MATE_SCORE)


def adjudicate_draw(white_hist, black_hist, adj: Adjudication) -> bool:
    if adj.draw_movenumber is None:
        return False
    if len(white_hist) + len(black_hist) < adj.draw_movenumber * 2:
        return False
    mc = adj.draw_movecount
    if min(len(white_hist), len(black_hist)) < mc:
        return False
    for score in white_hist[-mc:] + black_hist[-mc:]:
        if score is None or abs(score) > adj.draw_score:
            return False
    return True


def adjudicate_resign(white_hist, black_hist, adj: Adjudication):
    """Return the losing color, or None. Two-sided: both engines must agree."""
    if adj.resign_movecount is None or adj.resign_score is None:
        return None
    mc = adj.resign_movecount
    if min(len(white_hist), len(black_hist)) < mc:
        return None
    w, b = white_hist[-mc:], black_hist[-mc:]
    if any(s is None for s in w + b):
        return None
    if all(ws <= -adj.resign_score for ws in w) and all(bs >= adj.resign_score for bs in b):
        return chess.WHITE
    if all(bs <= -adj.resign_score for bs in b) and all(ws >= adj.resign_score for ws in w):
        return chess.BLACK
    return None


class EngineSlot:
    """One engine process owned by one worker thread, restartable on crash."""

    def __init__(self, config: EngineConfig, match_config: MatchConfig):
        self.config = config
        self.match_config = match_config
        self.engine = None
        self.needs_restart = False
        self.timed_out = False
        self._restarts = 0

    def start(self):
        cmd_path = Path(self.config.cmd)
        if cmd_path.is_file():
            # Absolute path before setting cwd, or a relative engine path
            # would be resolved against the engine folder itself.
            cmd = str(cmd_path.resolve())
            cwd = self.config.cwd or str(cmd_path.resolve().parent)
        else:
            cmd = self.config.cmd      # engine on PATH
            cwd = self.config.cwd

        popen = (chess.engine.SimpleEngine.popen_uci if self.config.proto == 'uci'
                 else chess.engine.SimpleEngine.popen_xboard)
        try:
            self.engine = popen(cmd, cwd=cwd)
        except Exception as err:
            raise MatchError(
                f'cannot start {self.config.name} engine "{self.config.cmd}": {err}') from err

        self._configure_options()

        # xboard engines can only limit by nodes when they announce feature nps.
        if (self.match_config.nodes is not None and self.config.proto == 'xboard'
                and not getattr(self.engine.protocol, 'features', {}).get('nps')):
            self.close()
            raise MatchError(
                f'{self.config.name} xboard engine has no "feature nps", so --nodes '
                f'is not possible; use --depth or a time control instead.')

        self.needs_restart = False

    def _configure_options(self):
        opts = {}
        for name, value in self.config.options.items():
            if name not in self.engine.options:
                _warn_once(f'{self.config.name}: engine has no option "{name}", skipped.')
                continue
            if self.engine.options[name].is_managed():
                _warn_once(f'{self.config.name}: option "{name}" is protocol-managed, skipped.')
                continue
            opts[name] = value
        if opts:
            self.engine.configure(opts)

    def restart(self):
        self._restarts += 1
        if self._restarts > self.match_config.max_restarts:
            raise MatchError(
                f'{self.config.name} engine crashed more than '
                f'{self.match_config.max_restarts} times, aborting the match.')
        logger.warning(f'restarting {self.config.name} engine '
                       f'(restart {self._restarts}/{self.match_config.max_restarts}) ...')
        self.close()
        self.start()

    def play(self, board, limit, allowance, game_id):
        """Ask for a move, killing the engine if it exceeds `allowance` sec.

        python-chess applies no timeout of its own to play() with depth/nodes/
        clock limits, so a hung engine would block forever without this.
        """
        self.timed_out = False

        def _kill():
            self.timed_out = True
            try:
                self.engine.close()
            except Exception:
                pass

        watchdog = threading.Timer(allowance, _kill)
        watchdog.start()
        try:
            return self.engine.play(board, limit, game=game_id,
                                    info=chess.engine.INFO_SCORE)
        finally:
            watchdog.cancel()

    def close(self):
        if self.engine is None:
            return
        try:
            self.engine.quit()
        except Exception:
            pass
        try:
            self.engine.close()
        except Exception:
            pass
        self.engine = None


def play_game(white_slot, black_slot, start_fen, cfg: MatchConfig,
              round_no, game_no, test_is_white) -> GameRecord:
    board = chess.Board(start_fen)
    tc_mode = cfg.depth is None and cfg.nodes is None
    clocks = {chess.WHITE: float(cfg.base_time_sec),
              chess.BLACK: float(cfg.base_time_sec)}
    inc = float(cfg.inc_time_sec)
    hist = {chess.WHITE: [], chess.BLACK: []}
    game_id = object()                 # fresh token -> engines get a newgame
    result, termination = None, 'normal'

    def mover_loses(mover):
        return '0-1' if mover == chess.WHITE else '1-0'

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            result = outcome.result()
            break

        if len(board.move_stack) >= MAX_PLIES:
            result, termination = '1/2-1/2', 'unterminated'
            break

        if adjudicate_draw(hist[chess.WHITE], hist[chess.BLACK], cfg.adjudication):
            result, termination = '1/2-1/2', 'adjudication: draw'
            break

        loser = adjudicate_resign(hist[chess.WHITE], hist[chess.BLACK], cfg.adjudication)
        if loser is not None:
            result = '0-1' if loser == chess.WHITE else '1-0'
            termination = 'adjudication: resign'
            break

        mover = board.turn
        slot = white_slot if mover == chess.WHITE else black_slot

        if cfg.depth is not None:
            limit = chess.engine.Limit(depth=cfg.depth)
            allowance = cfg.move_timeout
        elif cfg.nodes is not None:
            limit = chess.engine.Limit(nodes=cfg.nodes)
            allowance = cfg.move_timeout
        else:
            limit = chess.engine.Limit(
                white_clock=max(0.01, clocks[chess.WHITE]),
                black_clock=max(0.01, clocks[chess.BLACK]),
                white_inc=inc, black_inc=inc)
            allowance = clocks[mover] + cfg.time_margin

        start = time.perf_counter()
        try:
            res = slot.play(board, limit, allowance, game_id)
        except chess.engine.EngineError as err:
            slot.needs_restart = True
            if slot.timed_out:
                termination = 'time forfeit' if tc_mode else 'stalled'
            elif isinstance(err, chess.engine.EngineTerminatedError):
                termination = 'abandoned'
            else:
                termination = 'illegal move'
            logger.warning(f'game {round_no}.{game_no}: {slot.config.name} engine '
                           f'{termination} ({err})')
            result = mover_loses(mover)
            break

        if res.resigned:
            result, termination = mover_loses(mover), 'resigned'
            break

        if tc_mode:
            clocks[mover] -= time.perf_counter() - start
            if clocks[mover] < 0:
                result, termination = mover_loses(mover), 'time forfeit'
                break
            clocks[mover] += inc

        if res.move is None:
            slot.needs_restart = True
            result, termination = mover_loses(mover), 'abandoned'
            break

        hist[mover].append(mover_score(res))
        board.push(res.move)

    test_score = {'1-0': 1.0, '1/2-1/2': 0.5, '0-1': 0.0}[result]
    if not test_is_white:
        test_score = 1.0 - test_score

    return GameRecord(round=round_no, game_no=game_no, test_is_white=test_is_white,
                      start_fen=start_fen, moves=list(board.move_stack),
                      result=result, test_score=test_score, termination=termination)


def build_pgn(record: GameRecord, cfg: MatchConfig) -> chess.pgn.Game:
    board = chess.Board(record.start_fen)
    for move in record.moves:
        board.push(move)
    game = chess.pgn.Game.from_board(board)   # sets FEN/SetUp when needed

    if cfg.depth is not None:
        tc = f'depth {cfg.depth}'
    elif cfg.nodes is not None:
        tc = f'nodes {cfg.nodes}'
    else:
        tc = f'{cfg.base_time_sec}+{cfg.inc_time_sec}'

    game.headers['Event'] = cfg.event
    game.headers['Site'] = '?'
    game.headers['Date'] = datetime.date.today().strftime('%Y.%m.%d')
    game.headers['Round'] = f'{record.round}.{record.game_no}'
    game.headers['White'] = cfg.test.name if record.test_is_white else cfg.base.name
    game.headers['Black'] = cfg.base.name if record.test_is_white else cfg.test.name
    game.headers['Result'] = record.result
    game.headers['TimeControl'] = tc
    game.headers['PlyCount'] = str(len(record.moves))
    game.headers['Termination'] = record.termination
    return game


class MatchRunner:
    def __init__(self, cfg: MatchConfig, echo=False):
        self.cfg = cfg
        self.echo = echo               # True: print score lines to stdout (CLI)
        self.wins = self.losses = self.draws = 0
        self.records = []
        self._lock = threading.Lock()
        self._pgn_lock = threading.Lock()
        self._tls = threading.local()
        self._slots = []
        self._slots_lock = threading.Lock()
        self._abort = threading.Event()

    def run(self) -> MatchResult:
        self._validate()
        fens = self._select_openings()
        workers = max(1, min(self.cfg.concurrency, len(fens)))

        executor = ThreadPoolExecutor(max_workers=workers,
                                      thread_name_prefix='pymatch')
        futures = [executor.submit(self._worker, i + 1, fen)
                   for i, fen in enumerate(fens)]
        try:
            for future in as_completed(futures):
                future.result()        # propagate MatchError etc.
        except BaseException:
            # Fatal error or KeyboardInterrupt: drop queued rounds, kill the
            # engines so running games unwind, then re-raise.
            self._abort.set()
            for future in futures:
                future.cancel()
            self._close_all()
            raise
        finally:
            executor.shutdown(wait=True)
            self._close_all()

        self.records.sort(key=lambda r: (r.round, r.game_no))
        return MatchResult(wins=self.wins, losses=self.losses, draws=self.draws,
                           games=self.wins + self.losses + self.draws,
                           records=self.records)

    def _validate(self):
        cfg = self.cfg
        if cfg.games < 2 or cfg.games % 2 != 0:
            raise MatchError(f'games must be an even number >= 2, got {cfg.games}.')
        if cfg.depth is not None and cfg.nodes is not None:
            raise MatchError('define only one of depth and nodes.')
        if cfg.openings is None and cfg.opening_file is None:
            raise MatchError('no openings: set opening_file or pass an openings list.')
        if cfg.opening_format not in ('epd', 'pgn'):
            raise MatchError(f'unsupported opening format "{cfg.opening_format}".')
        if cfg.opening_order not in ('random', 'sequential'):
            raise MatchError(f'unsupported opening order "{cfg.opening_order}".')

    def _select_openings(self) -> List[str]:
        """Pick one start position per round. Only the picked lines are parsed
        into FENs; the pool itself stays as raw lines (see load_openings)."""
        rounds = self.cfg.games // 2

        if self.cfg.openings is not None:
            pool = self.cfg.openings
            if len(pool) < rounds:
                _warn_once(f'only {len(pool)} openings for {rounds} rounds; '
                           f'openings will repeat.')
            picked = [pool[i % len(pool)] for i in range(rounds)]
        else:
            pool = load_openings(self.cfg.opening_file, self.cfg.opening_format)
            if self.cfg.opening_order == 'sequential':
                picked = [pool[i % len(pool)] for i in range(rounds)]
            else:
                picked = [random.choice(pool) for _ in range(rounds)]

        fens = []
        for line in picked:
            try:
                fens.append(board_from_line(line).fen())
            except ValueError as err:
                raise MatchError(f'bad opening "{line[:60]}": {err}')
        return fens

    def _get_slots(self):
        if not hasattr(self._tls, 'pair'):
            test_slot = EngineSlot(self.cfg.test, self.cfg)
            base_slot = EngineSlot(self.cfg.base, self.cfg)
            test_slot.start()
            base_slot.start()
            with self._slots_lock:
                self._slots += [test_slot, base_slot]
            self._tls.pair = (test_slot, base_slot)
        return self._tls.pair

    def _worker(self, round_no, fen):
        test_slot, base_slot = self._get_slots()
        for game_no in (1, 2):
            if self._abort.is_set():
                return
            for slot in (test_slot, base_slot):
                if slot.needs_restart:
                    slot.restart()
            test_is_white = game_no == 1
            white, black = ((test_slot, base_slot) if test_is_white
                            else (base_slot, test_slot))
            record = play_game(white, black, fen, self.cfg,
                               round_no, game_no, test_is_white)
            self._record(record)

    def _record(self, record: GameRecord):
        with self._lock:
            if record.test_score == 1.0:
                self.wins += 1
            elif record.test_score == 0.0:
                self.losses += 1
            else:
                self.draws += 1
            w, l, d = self.wins, self.losses, self.draws
            self.records.append(record)

        games = w + l + d
        rate = (w + d / 2) / games
        white = self.cfg.test.name if record.test_is_white else self.cfg.base.name
        black = self.cfg.base.name if record.test_is_white else self.cfg.test.name
        line1 = (f'Finished game {games} ({white} vs {black}):'
                 f' {record.result} {{{record.termination}}}')
        line2 = (f'Score of {self.cfg.test.name} vs {self.cfg.base.name}:'
                 f' {w} - {l} - {d}  [{rate:.3f}] {games}')
        if self.echo:
            print(line1)
            print(line2, flush=True)
        else:
            logger.debug(line1)
            logger.debug(line2)

        if self.cfg.pgnout:
            try:
                game = build_pgn(record, self.cfg)
                with self._pgn_lock:
                    with open(self.cfg.pgnout, 'a', encoding='utf-8') as f:
                        print(game, file=f, end='\n\n')
            except OSError as err:
                _warn_once(f'cannot write pgn file {self.cfg.pgnout}: {err}')

    def _close_all(self):
        with self._slots_lock:
            slots, self._slots = self._slots, []
        for slot in slots:
            slot.close()


def run_match(cfg: MatchConfig, echo=False) -> MatchResult:
    """Play the configured match and return the result, test engine POV."""
    return MatchRunner(cfg, echo=echo).run()


def _parse_options(pairs):
    options = {}
    for pair in pairs or []:
        name, sep, value = pair.partition('=')
        if not sep:
            raise MatchError(f'bad option "{pair}", expected Name=Value.')
        options[name] = value
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=f'pymatch {__version__}',
        description='Engine vs engine match manager for standard chess, '
                    'UCI and CECP/xboard.')
    parser.add_argument('--test-engine', required=True, help='test engine file.')
    parser.add_argument('--base-engine', required=True, help='base engine file.')
    parser.add_argument('--test-name', default='test')
    parser.add_argument('--base-name', default='base')
    parser.add_argument('--protocol', default='uci', choices=('uci', 'cecp', 'xboard'),
                        help='protocol of both engines, default=uci.')
    parser.add_argument('--test-option', action='append', metavar='NAME=VALUE',
                        help='option for the test engine, repeatable.')
    parser.add_argument('--base-option', action='append', metavar='NAME=VALUE',
                        help='option for the base engine, repeatable.')
    parser.add_argument('--common-option', action='append', metavar='NAME=VALUE',
                        help='option for both engines, repeatable.')
    parser.add_argument('--games', type=int, default=2,
                        help='even number of games, 2 per opening, default=2.')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--depth', type=int, help='fixed depth per move.')
    parser.add_argument('--nodes', type=int, help='fixed nodes per move.')
    parser.add_argument('--base-time-sec', type=float, default=5.0)
    parser.add_argument('--inc-time-sec', type=float, default=0.05)
    parser.add_argument('--openings', required=True, metavar='FILE',
                        help='opening book: .epd/.fen (one per line) or .pgn.')
    parser.add_argument('--order', default='random', choices=('random', 'sequential'))
    parser.add_argument('--draw-movenumber', type=int)
    parser.add_argument('--draw-movecount', type=int, default=6)
    parser.add_argument('--draw-score', type=int, default=0)
    parser.add_argument('--resign-movecount', type=int)
    parser.add_argument('--resign-score', type=int)
    parser.add_argument('--pgnout', help='append played games to this pgn file.')
    parser.add_argument('--event', default='pymatch')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    suffix = Path(args.openings).suffix.lower()
    opening_format = 'pgn' if suffix == '.pgn' else 'epd'

    common = _parse_options(args.common_option)
    cfg = MatchConfig(
        test=EngineConfig(cmd=args.test_engine, name=args.test_name,
                          proto=args.protocol,
                          options={**common, **_parse_options(args.test_option)}),
        base=EngineConfig(cmd=args.base_engine, name=args.base_name,
                          proto=args.protocol,
                          options={**common, **_parse_options(args.base_option)}),
        games=args.games,
        base_time_sec=args.base_time_sec, inc_time_sec=args.inc_time_sec,
        depth=args.depth, nodes=args.nodes,
        opening_file=args.openings, opening_format=opening_format,
        opening_order=args.order,
        adjudication=Adjudication(
            draw_movenumber=args.draw_movenumber,
            draw_movecount=args.draw_movecount,
            draw_score=args.draw_score,
            resign_movecount=args.resign_movecount,
            resign_score=args.resign_score),
        pgnout=args.pgnout, concurrency=args.concurrency, event=args.event)

    result = run_match(cfg, echo=True)
    print('Finished match')
    return 0 if result.games == args.games else 1


if __name__ == '__main__':
    sys.exit(main())
