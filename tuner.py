#!/usr/bin/env python


"""Optuna Game Parameter Tuner

Game parameter tuner using optuna framework. The game can be a chess or
chess variants. Parameters can be piece values for evaluations or
futility pruning margin for search."""


__author__ = 'fsmosca'
__script_name__ = 'Optuna Game Parameter Tuner'
__version__ = 'v8.0.0'
__credits__ = ['joergoster', 'musketeerchess', 'optuna']


import sys
import os
import time
import random
import tempfile
from subprocess import Popen, PIPE
import copy
from collections import OrderedDict
import argparse
from pathlib import Path
import ast
import re
from typing import Tuple, Union
import logging
import math

import optuna
from optuna.distributions import IntDistribution, FloatDistribution


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.DEBUG)

fh = logging.FileHandler(filename='log_tuner.txt', mode='a')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-5.5s | %(message)s'))
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)


is_panda_ok = True
try:
    import pandas
    pandas.set_option("display.max_rows", None, "display.max_columns", None)
except ModuleNotFoundError:
    is_panda_ok = False
    logger.info('Warning! pandas is not installed.')


class Elo:
    """
    Ref.: https://github.com/cutechess/cutechess/blob/master/projects/lib/src/elo.cpp
    """
    def __init__(self, win, loss, draw):
        self.wins = win
        self.losses = loss
        self.draws = draw
        self.n = win + loss + draw
        self.mu = (self.wins / self.n + self.draws / self.n / 2) if self.n else 0.0

    def stdev(self):
        if self.n == 0:
            return 0.0
        n = self.n
        wr = self.wins / n
        lr = self.losses / n
        dr = self.draws / n

        dev_w = wr * math.pow(1.0 - self.mu, 2.0)
        dev_l = lr * math.pow(0.0 - self.mu, 2.0)
        dev_d = dr * math.pow(0.5 - self.mu, 2.0)

        return math.sqrt(dev_w + dev_l + dev_d) / math.sqrt(n)

    def draw_ratio(self):
        return self.draws / self.n if self.n else 0.0

    def diff(self, p=None):
        """Elo difference"""
        p = self.mu if p is None else p

        # Manage extreme values of p, if 1.0 or more make it 0.99.
        # If 0 or below make it 0.01. With 0.01 the The max rating diff is 800.
        p = min(0.99, max(0.01, p))
        return -400.0 * math.log10(1.0 / p - 1.0)

    def error_margin(self, confidence_level=95):
        a = (1 - confidence_level/100) / 2
        mu_min = self.mu + self.phi_inv(a) * self.stdev()
        mu_max = self.mu + self.phi_inv(1-a) * self.stdev()
        return (self.diff(mu_max) - self.diff(mu_min)) / 2.0

    def erf_inv(self, x):
        pi = 3.1415926535897

        a = 8.0 * (pi - 3.0) / (3.0 * pi * (4.0 - pi))
        y = math.log(1.0 - x * x)
        z = 2.0 / (pi * a) + y / 2.0

        ret = math.sqrt(math.sqrt(z * z - y / a) - z)

        if x < 0.0:
            return -ret
        return ret

    def phi_inv(self, p):
        return math.sqrt(2.0) * self.erf_inv(2.0 * p - 1.0)

    def los(self):
        """LOS - Likelihood Of Superiority"""
        if self.wins == 0 and self.losses == 0:
            return 0
        return 100 * (0.5 + 0.5 * math.erf((self.wins - self.losses) / math.sqrt(2.0 * (self.wins + self.losses))))

    def confidence_interval(self, confidence_level=95, type_='elo'):
        e = self.diff()
        em = self.error_margin(confidence_level)

        if type_ == 'rate':
            return self.expected_score_rate(e-em), self.expected_score_rate(e+em)
        else:
            return e-em, e+em

    def expected_score_rate(self, rd):
        return 1 / (1 + 10 ** (-rd/400))


# Match-manager score line, e.g. "Score of test vs base: 39 - 28 - 64  [0.542] 131".
SCORE_LINE_RE = re.compile(r': (\d+) - (\d+) - (\d+)\s*\[[0-9.]+\]\s*(\d+)\s*$')


def option_tokens(params: dict) -> list:
    """Engine option argv tokens, e.g. ['option.pawn=92', 'option.knight=300']."""
    return [f'option.{k}={v}' for k, v in params.items()]


def options_str(params: dict) -> str:
    """Space-joined option_tokens, for logging."""
    return ' '.join(option_tokens(params))


def tokens_to_dict(args_tokens) -> dict:
    """
    Flatten CLI tokens like [['name=tpe', 'seed=1']] into {'name': 'tpe',
    'seed': '1'}. Later tokens win when a key repeats.
    """
    d = {}
    for opt in args_tokens or []:
        for tok in opt:
            if '=' in tok:
                k, _, v = tok.partition('=')
                d[k] = v
    return d


class Objective(object):
    def __init__(self, study, engine, input_param,
                 init_param, init_value, variant,
                 opening_file,
                 opening_format, pgnout,
                 nodes: Union[None, int]=None, base_time_sec=5,
                 inc_time_sec=0.05,
                 concurrency=1, proto='uci',
                 match_manager='pymatch', match_manager_file=None,
                 depth: Union[None, int]=None, games_per_trial=32,
                 threshold_pruner={}, common_param=None,
                 resign_movecount=None, resign_score=None,
                 draw_movenumber=None, draw_movecount=6, draw_score=0,
                 opening_posperfile=-1, n_startup_trials=1,
                 noisy_result=False, elo_objective=False,
                 use_affinity=None):
        self.study = study
        self.input_param = copy.deepcopy(input_param)
        self.init_param = copy.deepcopy(init_param)
        self.init_value = init_value

        self.variant = variant

        self.test_name = 'test'
        self.base_name = 'base'

        self.e1 = engine
        self.e2 = engine

        self.concurrency = concurrency
        self.opening_file = opening_file
        self.opening_format = opening_format
        # Cache of opening lines (epd/fen), loaded lazily for the pruner's
        # per-trial unique-opening sampling; stays None until first needed.
        self._opening_pool = None
        self.pgnout = pgnout
        self.nodes = nodes

        self.base_time_sec = base_time_sec
        self.inc_time_sec = inc_time_sec

        self.test_param = {}
        self.proto = proto

        self.match_manager = match_manager
        self.match_manager_file = match_manager_file
        self.depth = depth
        self.games_per_trial = games_per_trial

        self.startup_trials = n_startup_trials
        self.threshold_pruner = copy.deepcopy(threshold_pruner)

        if self.match_manager in ('pymatch', 'cutechess') and self.proto == 'cecp':
            self.proto = 'xboard'

        # fastchess only speaks UCI.
        if self.match_manager == 'fastchess':
            self.proto = 'uci'

        # fastchess thread affinity: None=off, True=bare flag, str=core list/range.
        self.use_affinity = use_affinity

        self.common_param = common_param
        self.resign_movecount = resign_movecount
        self.resign_score = resign_score
        self.draw_movenumber = draw_movenumber
        self.draw_movecount = draw_movecount
        self.draw_score = draw_score
        self.opening_posperfile = opening_posperfile
        self.trial_hist_check, self.pruned_hist = self.save_trial_history()
        self.noisy_result = noisy_result
        self.elo_objective = elo_objective
        self.value_name = 'Elo' if self.elo_objective else 'rate'

    def save_trial_history(self):
        """
        Rebuild the duplicate-suggestion caches from the study history.

        Returns (values, pruned): completed params mapped to their objective
        value, and the set of params whose trial was pruned. A pruned trial's
        value is only a partial-match result, so we remember that the param
        was pruned rather than the number itself.
        """
        values, pruned = {}, set()
        for t in self.study.trials:
            # Don't include interrupted trials.
            if t.state == optuna.trial.TrialState.RUNNING:
                continue

            trial_hist = {}
            for k, v in t.params.items():
                # Round floats to 5 dp so the key matches the one built in
                # __call__ (which rounds suggest_float to 5 dp); otherwise
                # float noise makes the same suggestion look like two keys.
                if isinstance(t.distributions.get(k), FloatDistribution):
                    v = round(v, 5)
                trial_hist[k] = v

            key = self.gen_parval_key(trial_hist)

            if t.state == optuna.trial.TrialState.PRUNED:
                pruned.add(key)
            elif t.value is not None:
                # FAILed trials have no value; caching None here would later
                # be returned as an objective value.
                values[key] = t.value

        return values, pruned

    def read_result(self, line: str) -> Tuple[float, int, int, int, int]:
        """
        Read result output line from match manager.

        # Score of e1 vs e2: 39 - 28 - 64  [0.542] 131
        wins=39, losses=28, draws=64, games=131
        """
        m = SCORE_LINE_RE.search(line)
        if m is None:
            raise ValueError(f'Unexpected score line from match manager: "{line}"')
        num_wins, num_losses, num_draws, num_games = map(int, m.groups())
        result = (num_wins + num_draws / 2) / num_games

        return result, num_wins, num_losses, num_draws, num_games

    @staticmethod
    def set_param(input_param):
        """
        Create a dict of default values from input param
        with default, min, max, and step.
        """
        return {k: v['default'] for k, v in input_param.items()}

    def get_match_commands(self, test_options, base_options, games, opening_file_override=None):
        # cutechess-cli and fastchess share most of the command syntax; duel.py
        # is the remaining branch. fastchess is handled as a cutechess-style CLI
        # with a few flag differences (see notes below).
        cutechess_style = self.match_manager in ('cutechess', 'fastchess')

        # Resolve the match-manager executable into argv form.
        if self.match_manager == 'cutechess':
            if self.match_manager_file:
                # User-supplied path wins (e.g. a self-compiled cutechess-cli on Linux).
                cmd = [self.match_manager_file]
            else:
                exe_name = 'cutechess-cli.exe' if sys.platform == 'win32' else 'cutechess-cli'
                bundled = Path(Path.cwd(), 'tourney_manager', 'cutechess', exe_name)
                # The repo ships a Windows cutechess-cli.exe; use it when present.
                # On Linux fall back to a cutechess-cli found on PATH.
                cmd = [str(bundled) if bundled.is_file() else exe_name]
        elif self.match_manager == 'fastchess':
            if self.match_manager_file:
                cmd = [self.match_manager_file]
            else:
                exe_name = 'fastchess.exe' if sys.platform == 'win32' else 'fastchess'
                bundled = Path(Path.cwd(), 'tourney_manager', 'fastchess', exe_name)
                # No fastchess binary ships with the repo; use one if the user
                # dropped it here, otherwise fall back to fastchess on PATH.
                cmd = [str(bundled) if bundled.is_file() else exe_name]
        else:
            # duel.py is a bundled Python script run via the interpreter; the
            # override lets users point at a duel.py in a custom location.
            duel_script = self.match_manager_file or './tourney_manager/duel/duel.py'
            cmd = [sys.executable, '-u', duel_script]

        cmd += ['-concurrency', str(self.concurrency)]

        if cutechess_style:
            # fastchess treats a bare relative cmd (e.g. engines/sf.exe) as a
            # PATH lookup and fails, so give it an absolute engine path.
            if self.match_manager == 'fastchess':
                e1 = str(Path(self.e1).resolve())
                e2 = str(Path(self.e2).resolve())
            else:
                e1, e2 = self.e1, self.e2
            cmd += ['-engine', f'cmd={e1}', f'name={self.test_name}', *test_options, f'proto={self.proto}']
            cmd += ['-engine', f'cmd={e2}', f'name={self.base_name}', *base_options, f'proto={self.proto}']
            if self.match_manager == 'cutechess':
                cmd += ['-wait', '100']
        else:
            cmd += ['-engine', f'cmd={self.e1}', f'name={self.test_name}', *test_options]
            cmd += ['-engine', f'cmd={self.e2}', f'name={self.base_name}', *base_options]

        if self.variant != 'normal':
            cmd += ['-variant', self.variant]

        # fastchess spells round-robin without the hyphen.
        cmd += ['-tournament',
                'roundrobin' if self.match_manager == 'fastchess' else 'round-robin']

        if cutechess_style:
            cmd += ['-recover', '-rounds', str(games//2), '-games', '2', '-repeat', '2']

            # fastchess takes a fixed depth/nodes directly; cutechess needs tc=inf.
            tc_inf = [] if self.match_manager == 'fastchess' else ['tc=inf']
            if self.depth is not None:
                cmd += ['-each', *tc_inf, f'depth={self.depth}']
            elif self.nodes is not None:
                cmd += ['-each', *tc_inf, f'nodes={self.nodes}']
            else:
                cmd += ['-each', f'tc=0/0:{self.base_time_sec}+{self.inc_time_sec}']
        # duel.py match manager
        else:
            cmd += ['-rounds', str(games//2), '-repeat', '2']
            if self.depth is not None:
                cmd += ['-each', 'tc=inf', f'depth={self.depth}']
            else:
                cmd += ['-each', f'tc=0/0:{self.base_time_sec}+{self.inc_time_sec}']

        if cutechess_style:
            if opening_file_override:
                # Pruner mode: play this batch's pre-sampled, trial-unique slice
                # in file order so openings never repeat within a trial.
                cmd += ['-openings', f'file={opening_file_override}',
                        'order=sequential', f'format={self.opening_format}']
            else:
                cmd += ['-openings', f'file={self.opening_file}',
                        'order=random', f'format={self.opening_format}']
        else:
            cmd += ['-openings', f'file={self.opening_file}',
                    f'posperfile={self.opening_posperfile}']

        # draw adjudication
        if self.draw_movenumber is not None:
            cmd += ['-draw', f'movenumber={self.draw_movenumber}',
                    f'movecount={self.draw_movecount}', f'score={self.draw_score}']

        if self.resign_movecount is not None and self.resign_score is not None:
            cmd += ['-resign', f'movecount={self.resign_movecount}', f'score={self.resign_score}']
            if cutechess_style:
                cmd.append('twosided=true')

        # fastchess uses the file= form and must emit cutechess-compatible output
        # so read_result can parse the score lines.
        if self.match_manager == 'fastchess':
            cmd += ['-pgnout', f'file={self.pgnout}', '-output', 'format=cutechess']
            # Thread affinity: bare flag auto-binds, otherwise a core list/range.
            if self.use_affinity is not None:
                cmd.append('-use-affinity')
                if self.use_affinity is not True:
                    cmd.append(str(self.use_affinity))
        else:
            cmd += ['-pgnout', self.pgnout]

        return cmd

    def engine_match(self, test_params, base_params, games=50, opening_override=None) -> Tuple[float, int, int, int, int]:
        """
        Play a match of `games` games and return (rate, wins, losses, draws,
        games) from the test engine's POV. test_params/base_params are the
        full engine option dicts (tuned/init params merged with common param).
        opening_override carries a pruner batch: a fen/epd line list for the
        built-in pymatch manager, or a temp file path for the external ones.
        """
        requested_games = games

        if self.match_manager == 'pymatch':
            result, wins, losses, draws, games = self._pymatch_match(
                test_params, base_params, games, opening_override)
        else:
            result, wins, losses, draws, games = self._external_match(
                option_tokens(test_params), option_tokens(base_params),
                games, opening_override)

        # e.g. the match manager died mid-match; don't accept a partial score silently.
        if games != requested_games:
            logger.warning(f'engine match played {games} of the requested '
                           f'{requested_games} games; using the partial result.')

        return result, wins, losses, draws, games

    def _pymatch_match(self, test_params, base_params, games, openings):
        # Only the built-in manager needs the python-chess dependency,
        # so it is imported here rather than at module level.
        import pymatch

        adjudication = pymatch.Adjudication(
            draw_movenumber=int(self.draw_movenumber) if self.draw_movenumber is not None else None,
            draw_movecount=int(self.draw_movecount),
            draw_score=int(self.draw_score),
            resign_movecount=int(self.resign_movecount) if self.resign_movecount is not None else None,
            resign_score=int(self.resign_score) if self.resign_score is not None else None)

        cfg = pymatch.MatchConfig(
            test=pymatch.EngineConfig(cmd=self.e1, name=self.test_name,
                                      proto=self.proto, options=test_params),
            base=pymatch.EngineConfig(cmd=self.e2, name=self.base_name,
                                      proto=self.proto, options=base_params),
            games=games,
            base_time_sec=self.base_time_sec, inc_time_sec=self.inc_time_sec,
            depth=self.depth, nodes=self.nodes,
            opening_file=self.opening_file, opening_format=self.opening_format,
            opening_order='random', openings=openings,
            adjudication=adjudication, pgnout=self.pgnout,
            concurrency=self.concurrency, event=__script_name__)

        res = pymatch.run_match(cfg)
        return res.rate, res.wins, res.losses, res.draws, res.games

    def _external_match(self, test_options, base_options, games, opening_file_override):
        result = ''
        wins, losses, draws = 0, 0, 0

        command = self.get_match_commands(
            test_options, base_options, games, opening_file_override)

        # An argv list needs no quoting/tokenizing: Popen passes each element
        # through as one argument, even if it contains spaces.
        process = Popen(command, stdout=PIPE, text=True)
        try:
            for eline in iter(process.stdout.readline, ''):
                line = eline.strip()
                if line.startswith(f'Score of {self.test_name} vs {self.base_name}'):
                    result, wins, losses, draws, games = self.read_result(line)
                if 'Finished match' in line:
                    break
        finally:
            # Closing stdout lets a still-running child fail its writes (EPIPE)
            # and exit; wait() reaps it so we never leak or zombie the process.
            if process.stdout is not None:
                process.stdout.close()
            process.wait()

        if result == '':
            raise Exception('Error, there is something wrong with the engine match.')

        return result, wins, losses, draws, games

    def sample_trial_openings(self):
        """
        Draw a list of unique openings for the current trial at random from the
        whole opening file. The pruner hands each batch a disjoint slice of this
        list, so no opening repeats within a trial (duplicates across trials are
        fine). Returns None to fall back to the match manager's own
        randomization for non line-based books (e.g. pgn) or when the file has
        too few positions to fill a trial without repeats.
        """
        if self.opening_format != 'epd':
            return None

        if self._opening_pool is None:
            try:
                with open(self.opening_file) as f:
                    self._opening_pool = [ln.strip() for ln in f if ln.strip()]
            except OSError as e:
                logger.warning(f'unique-opening sampling disabled, cannot read '
                               f'opening file "{self.opening_file}": {e}')
                self._opening_pool = []

        # The pruner's first batch plays `games` outright (only later batches are
        # clamped to games_per_trial), so a trial can consume up to
        # max(games_per_trial, games) games, i.e. that many // 2 openings.
        batch_games = self.threshold_pruner.get('games', 0)
        need = max(self.games_per_trial, batch_games) // 2
        if len(self._opening_pool) < need:
            logger.warning(
                f'opening file has {len(self._opening_pool)} positions but the '
                f'trial needs {need} unique openings; falling back to random '
                f'openings (they may repeat within the trial).')
            return None

        return random.sample(self._opening_pool, need)

    def write_batch_openings(self, openings, trial_number):
        """
        Write a batch's opening slice to a temp file (cutechess/fastchess read it
        with order=sequential) and return its path. The caller removes the file
        once the match is done.
        """
        fd, path = tempfile.mkstemp(
            prefix=f'ogpt_trial{trial_number}_b', suffix=f'.{self.opening_format}')
        with os.fdopen(fd, 'w') as f:
            f.write('\n'.join(openings) + '\n')
        return path

    @staticmethod
    def get_sampler(args_sampler):
        if args_sampler is None:
            logger.warning('Sampler option is not defined, use tpe sampler.')
            return optuna.samplers.TPESampler(), 10

        opts = tokens_to_dict(args_sampler)
        name = opts.get('name')

        if name is None:
            logger.warning('Sampler name is not defined, use tpe sampler.')
            return optuna.samplers.TPESampler(), 10

        seed = int(opts['seed']) if 'seed' in opts else None

        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
        if name == 'tpe':
            n_startup_trials = int(opts.get('n_startup_trials', 10))
            multivariate = opts.get('multivariate', 'false').lower() == 'true'
            group = opts.get('group', 'false').lower() == 'true'

            # Avoid ValueError when multivariate is false and group is true
            if group and not multivariate:
                group = False
                logger.warning('group is set to false as multivariate is false!')

            return optuna.samplers.TPESampler(
                n_ei_candidates=int(opts.get('n_ei_candidates', 24)),
                multivariate=multivariate, group=group, seed=seed,
                constant_liar=opts.get('constant_liar', 'false').lower() == 'true',
                n_startup_trials=n_startup_trials), n_startup_trials

        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html
        if name == 'cmaes':
            n_startup_trials = int(opts.get('n_startup_trials', 1))
            sigma0 = float(opts['sigma0']) if 'sigma0' in opts else None  # initial std. deviation

            return optuna.samplers.CmaEsSampler(
                sigma0=sigma0, seed=seed, n_startup_trials=n_startup_trials), n_startup_trials

        raise ValueError(f'sampler name "{name}" is not supported, use tpe or cmaes.')

    @staticmethod
    def get_pruner(args_threshold_pruner, args_hyperband_pruner, games_per_trial,
                   elo_objective=False, direction='maximize'):
        # Returns (optuna pruner object for create_study, th_pruner cadence dict).
        # th_pruner drives the batched match loop in __call__; its 'games'/
        # 'interval' set the report cadence for whichever pruner is active.
        pruner, th_pruner = None, {}
        minimize = direction == 'minimize'

        def _even(n):
            # A match batch needs an even game count (each opening is played
            # with both colors) and at least one 2-game round.
            n = max(2, int(n))
            return n if n % 2 == 0 else n + 1

        if args_hyperband_pruner is not None:
            # Hyperband meta-pruner. The loop reports the running result at
            # step=played_games, so the resource unit is "games played": it plays
            # `games` (=min_resource by default) per batch, reports each time, and
            # Hyperband prunes weak trials at its rungs.
            opts = tokens_to_dict(args_hyperband_pruner)
            min_resource = _even(opts.get('min_resource', games_per_trial // 4))
            max_resource = int(opts.get('max_resource', games_per_trial))
            reduction_factor = int(opts.get('reduction_factor', 3))
            bootstrap_count = int(opts.get('bootstrap_count', 0))
            interval = int(opts.get('interval', 1))

            th_pruner = {'name': 'hyperband', 'games': min_resource, 'interval': interval}

            logger.info(f'pruner name: hyperband,'
                        f' min_resource: {min_resource},'
                        f' max_resource: {max_resource},'
                        f' reduction_factor: {reduction_factor},'
                        f' bootstrap_count: {bootstrap_count},'
                        f' interval: {interval}\n')

            pruner = optuna.pruners.HyperbandPruner(
                min_resource=min_resource, max_resource=max_resource,
                reduction_factor=reduction_factor, bootstrap_count=bootstrap_count)

        elif args_threshold_pruner is not None:
            opts = tokens_to_dict(args_threshold_pruner)

            # Unlike hyperband, the threshold pruner has a fixed bound and no
            # notion of direction, so the bound has to match the search.
            # Maximizing (stronger params) prunes results below the threshold;
            # minimizing (deliberately weaker params) prunes results above it.
            # The defaults mirror each other around a tied match (0.5 / 0 Elo).
            if elo_objective:
                default_result = 10.0 if minimize else -10.0
            else:
                default_result = 0.75 if minimize else 0.25

            th_pruner = {
                'name': 'threshold',
                # Default result threshold is in the objective's unit.
                'result': float(opts.get('result', default_result)),
                'games': _even(opts.get('games', games_per_trial // 2)),
                'interval': int(opts.get('interval', 1)),
            }

            logger.info(f'pruner name: threshold_pruner,'
                        f' prune when result is'
                        f' {"above" if minimize else "below"} {th_pruner["result"]},'
                        f' games: {th_pruner["games"]},'
                        f' interval: {th_pruner["interval"]}\n')

            bound = ({'upper': th_pruner['result']} if minimize
                     else {'lower': th_pruner['result']})
            pruner = optuna.pruners.ThresholdPruner(
                n_warmup_steps=th_pruner["games"],
                interval_steps=th_pruner["interval"], **bound)

        return pruner, th_pruner

    def gen_parval_key(self, param):
        """
        Generates a key based on param name and param value.
        """
        return ','.join(f'{k}={v}' for k, v in param.items())

    def __call__(self, trial):
        logger.info('')
        logger.info(f'starting trial: {trial.number} ...')
        logger.info(f'deterministic function: {not self.noisy_result}')
        start_time = time.perf_counter()

        # Suggest values for the parameters to optimize.
        for k, v in self.input_param.items():
            if v.get('type') == 'float':
                par_val = round(trial.suggest_float(k, v['min'], v['max'], step=v['step']), 5)
            # Otherwise use integer.
            else:
                par_val = trial.suggest_int(k, v['min'], v['max'], step=v['step'])
            self.test_param[k] = par_val

        # Check if param values suggested by sampler was already suggested before and if so
        # just retrieve the objective value and send it to the sampler.
        test_param_key = self.gen_parval_key(self.test_param)
        was_pruned = test_param_key in self.pruned_hist
        if was_pruned or test_param_key in self.trial_hist_check:
            logger.warning(f'Duplicate suggestion from sampler, {self.test_param}')
            if self.noisy_result:
                logger.warning('Execute engine match as --noisy-result flag is enabled.')
            elif was_pruned:
                # Only a partial result exists for this param, from the match
                # that was pruned. Prune again instead of returning that value,
                # which would record a completed trial for games never played.
                logger.warning('Prune this trial, the param was pruned before.')
                elapse_time(time.perf_counter() - start_time)
                raise optuna.TrialPruned()
            else:
                value = self.trial_hist_check[test_param_key]
                logger.warning(f'Just return previous value of {value}')
                return value

        # Full engine option dicts. Common param goes to both engines and
        # should not be included in the test param (overlap is rejected at
        # startup). The base engine plays the init param.
        test_params = dict(self.test_param)
        base_params = dict(self.init_param)
        if self.common_param is not None:
            test_params.update(self.common_param)
            base_params.update(self.common_param)

        # Log info to console.
        logger.info(f'suggested param for test engine: {self.test_param}')
        logger.info(f'param for base engine          : {self.init_param}')

        if self.common_param is not None:
            logger.info(f'common param: {self.common_param}')

        logger.info(f'init param: {self.init_param}')
        logger.info(f'init objective value: {self.init_value}')

        # Read the best trial from optuna in one go: it already tracks the best
        # for the study's direction, and taking all three lines from the same
        # trial keeps them consistent with each other.
        best = self.study.best_trial
        logger.info(f'study best param: {best.params}')
        if self.elo_objective:
            logger.info(f'study best objective value: Elo {best.value}')
        else:
            logger.info(f'study best objective value: {best.value}')
        logger.info(f'study best trial number: {best.number}')

        # Run engine vs engine match.

        # Handle trial pruning if there is. We only play partial games instead of the full
        # games per trial. If the result is bad, we prune this trial thereby saving time.
        wins, losses, draws, games = 0, 0, 0, 0
        if (len(self.threshold_pruner)
                and len(self.study.trials) > self.startup_trials):
            games_to_play = self.threshold_pruner['games']
            result, played_games = 0.0, 0

            # Pre-sample this trial's openings so the pruner's batches (each a
            # separate match) never reuse the same opening within the trial.
            # Works with the managers that accept a custom opening set:
            # pymatch (direct list) and cutechess/fastchess (temp file played
            # sequentially); line-based (epd/fen) books only. Otherwise
            # trial_openings is None and behaviour is unchanged.
            supports_override = self.match_manager in ('pymatch', 'cutechess', 'fastchess')
            trial_openings = self.sample_trial_openings() if supports_override else None
            opening_cursor = 0

            while True:
                logger.info(f'games_to_play: {games_to_play} ...')

                # Give this batch its own disjoint slice of the trial's openings.
                batch_opening = None
                if trial_openings is not None:
                    n = games_to_play // 2
                    chunk = trial_openings[opening_cursor:opening_cursor + n]
                    opening_cursor += n
                    if self.match_manager == 'pymatch':
                        batch_opening = chunk
                    else:
                        batch_opening = self.write_batch_openings(chunk, trial.number)

                try:
                    cur_result, pwins, plosses, pdraws, pgames = self.engine_match(
                        test_params, base_params, games_to_play, batch_opening)
                finally:
                    if isinstance(batch_opening, str) and os.path.exists(batch_opening):
                        os.remove(batch_opening)

                wins += pwins; losses += plosses; draws += pdraws; games += pgames
                played_games += games_to_play

                # This batch's own result, and the cumulative result over all
                # games of the trial so far (every game weighs the same, no
                # matter how the batches were sized).
                if self.elo_objective:
                    cur_result = round(Elo(pwins, plosses, pdraws).diff(), 0)
                    result = round(Elo(wins, losses, draws).diff(), 0)
                else:
                    cur_result = round(cur_result, 5)
                    result = round((wins + draws / 2) / games, 5)

                logger.info(f'result: {{intermediate: {self.value_name} {cur_result}, G/W/D/L: {pgames}/{pwins}/{pdraws}/{plosses}}}')
                logger.info(f'result: {{cumulative: {self.value_name} {result}, G/W/D/L: {games}/{wins}/{draws}/{losses}}}')

                trial.report(result, played_games)

                if trial.should_prune():
                    # Remember that this param was pruned. Its partial result
                    # must not be reused as if it were a completed match.
                    self.pruned_hist.add(test_param_key)
                    elapse_time(time.perf_counter() - start_time)
                    raise optuna.TrialPruned()

                if played_games >= self.games_per_trial:
                    break

                games_to_play = min(
                    self.games_per_trial - played_games,
                    self.threshold_pruner['games'] * self.threshold_pruner['interval']
                )
        # Else if there is no trial pruner just proceed with normal game test at full games per trial.
        else:
            result, wins, losses, draws, games = self.engine_match(test_params, base_params, self.games_per_trial)

        result = round(result, 5)

        # Log elo diff and confidence interval.
        elo = Elo(wins, losses, draws)
        confidence_level = 95
        ci_low_pct, ci_high_pct = elo.confidence_interval(confidence_level=confidence_level, type_='rate')
        ci_low_elo, ci_high_elo = elo.confidence_interval(confidence_level=confidence_level, type_='elo')
        elodiff = elo.diff()
        em = elo.error_margin(confidence_level)
        los = elo.los()
        dr = elo.draw_ratio()

        if self.elo_objective:
            result = round(elodiff, 0)
            cilow = f'{ci_low_elo:+0.1f}'
            cihigh = f'{ci_high_elo:+0.1f}'
        else:
            cilow = f'{ci_low_pct:0.5f}'
            cihigh = f'{ci_high_pct:0.5f}'

        logger.info(f'Actual match result: {self.value_name} {result},'
                    f' CI: [{cilow}, {cihigh}], CL: {confidence_level}%,'
                    f' G/W/D/L: {games}/{wins}/{draws}/{losses}, POV: optimizer')

        logger.info(f'Elo Diff: {elodiff:+0.1f}, ErrMargin: +/- {em:0.1f},'
                    f' CI: [{ci_low_elo:+0.1f}, {ci_high_elo:+0.1f}],'
                    f' LOS: {los:0.1f}%, DrawRatio: {100 * dr:0.2f}%')

        # Output for match manager.
        logger.info(f'test param format for match manager: {options_str(self.test_param)}')

        # Cache the result for duplicate suggestions from the sampler. A full
        # match supersedes any earlier pruned partial for the same param.
        self.trial_hist_check[test_param_key] = result
        self.pruned_hist.discard(test_param_key)

        logger.info(f'result sent to optimizer: {result}')
        elapse_time(time.perf_counter() - start_time)

        return result


def save_plots(study, study_name, input_param, is_plot=False):
    if not is_plot:
        return

    logger.info('Saving plots ...')

    # Use optuna's matplotlib backend so plots are saved as static PNG images.
    import warnings
    try:
        import matplotlib
        matplotlib.use('Agg')  # render to a file, no display/browser needed
        import matplotlib.pyplot as plt
        from matplotlib.contour import ContourSet
        from optuna.visualization.matplotlib import (
            plot_optimization_history, plot_slice, plot_contour,
            plot_parallel_coordinate, plot_param_importances)
    except ModuleNotFoundError as err:
        logger.warning(f'Skipped plots, matplotlib is not installed: {err}')
        return

    trials = len(study.trials)
    bg = '#F7D0CA'
    params = list(input_param.keys())

    os.makedirs('visuals', exist_ok=True)
    pre_name = f'./visuals/{study_name}_{trials}'

    n = len(params)

    def save_current(filename, figsize=None, tight=True):
        fig = plt.gcf()
        # Long parameter names overlap on the default canvas, so grow the
        # figure with the number of parameters.
        if figsize is not None:
            fig.set_size_inches(*figsize)
        if tight:
            fig.savefig(filename, dpi=100, bbox_inches='tight', facecolor=bg)
        else:
            fig.savefig(filename, dpi=100, facecolor=bg)
        plt.close(fig)

    def try_plot(make_plot, filename, figsize=None, tight=True):
        # Plotting is best-effort: a degenerate study (too few trials, or zero
        # variance in objective values -> optuna's fANOVA raises RuntimeError)
        # must never crash an optimization run that already finished.
        try:
            make_plot()
            save_current(filename, figsize, tight=tight)
        except Exception as err:
            logger.warning(f'Skipped plot "{filename}": {err}')
            plt.close('all')

    def find_colorbar_mappable(panels):
        # Prefer the filled contour set (contour plot); fall back to the
        # indexed scatter (slice plot). This skips the black line contour, the
        # grey infeasible scatter and other non-mappable artists, so the
        # colorbar always shares the contour's Blues cmap instead of falling
        # back to the default viridis.
        for ax in panels:
            for coll in ax.collections:
                if isinstance(coll, ContourSet) and getattr(coll, 'filled', False):
                    return coll
        for ax in panels:
            for coll in ax.collections:
                try:
                    if coll.get_array() is not None:
                        return coll
                except Exception:
                    continue
        return None

    def attach_fixed_colorbar(axs, figsize, label):
        # optuna adds the colorbar via `fig.colorbar(mappable, ax=all_axes)`,
        # which steals space from every panel; combined with bbox_inches='tight'
        # this inflates the right margin and the colorbar width as the panel
        # count grows. Replace it with a fixed-position strip and lay the grid
        # out into a rect that reserves room for it, so the layout stays stable.
        # Used by the contour and slice plots (both build a multi-panel grid).
        panels = axs.ravel() if hasattr(axs, 'ravel') else [axs]
        fig = panels[0].figure
        fig.set_size_inches(*figsize)
        # Drop optuna's auto colorbar (the last axes it appended).
        if len(fig.axes) > len(panels):
            fig.axes[-1].remove()
        # Pack the grid into the left/center, reserving a right strip for the
        # colorbar and a top strip for the suptitle.
        fig.tight_layout(rect=[0, 0.005, 0.91, 0.965])
        mappable = find_colorbar_mappable(panels)
        if mappable is not None:
            cax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
            cb = fig.colorbar(mappable, cax=cax)
            cb.set_label(label)

    with warnings.catch_warnings():
        # Plotting is best-effort; silence its warnings (e.g. optuna's
        # ExperimentalWarning, matplotlib singular-limits on flat data).
        warnings.simplefilter('ignore')

        try_plot(lambda: plot_optimization_history(study),
                 f'{pre_name}_hist.png')
        slice_panel_width_inches = 5.0
        slice_height_inches = 6.0

        def plot_slice_fixed():
            axs = plot_slice(study, params=params)
            # Same multi-panel colorbar issue as the contour plot (optuna's
            # `fig.colorbar(sc, ax=axs)` steals from every panel). Rebuild it
            # on a fixed strip and keep each panel a readable width. The slice
            # colorbar encodes the trial number, hence the 'Trial' label.
            if hasattr(axs, 'ravel'):
                attach_fixed_colorbar(
                    axs,
                    (slice_panel_width_inches * n, slice_height_inches),
                    'Trial')

        try_plot(plot_slice_fixed,
                 f'{pre_name}_slice.png', tight=False)
        contour_cell_inches = 5.0

        def plot_contour_fixed():
            axs = plot_contour(study, params=params)
            # Only the n>2 case lays out an n*n grid and needs the colorbar
            # rebuild; n<=2 is a single panel optuna already lays out cleanly.
            if hasattr(axs, 'ravel'):
                attach_fixed_colorbar(
                    axs,
                    (contour_cell_inches * n, contour_cell_inches * n),
                    'Objective Value')

        try_plot(plot_contour_fixed,
                 f'{pre_name}_contour.png', tight=False)
        try_plot(lambda: plot_parallel_coordinate(study, params=params),
                 f'{pre_name}_parallel.png',
                 figsize=(max(10, 3.5 * n), 6))
        def plot_importances_5dp():
            # optuna labels each importance bar to 2 decimals (and "<0.01" for
            # tiny values). Each label is drawn at an x position equal to the
            # true importance value, so rewrite the text to 5 decimal places
            # without recomputing importances.
            ax = plot_param_importances(study)
            for text in ax.texts:
                text.set_text(f'{text.get_position()[0]:.5f}')

        # One horizontal bar per param: grow the height with n so bars stay a
        # readable thickness instead of thinning out as the param count grows.
        try_plot(plot_importances_5dp, f'{pre_name}_importance.png',
                 figsize=(9, max(5.0, 0.6 * n + 2.0)))

    logger.info('Done saving plots.\n')


def save_study_log(study, study_name, elo_objective, direction='maximize'):
    if not is_panda_ok:
        return

    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    logger.info(f'{df.to_string(index=False)}\n')
    df.to_csv(f'{study_name}.csv', index=False)

    # Show a frame where same param are grouped and value is averaged.
    # Pruned trials carry a partial result over fewer games; averaging them
    # with completed trials would bias the mean, so keep COMPLETE only.
    df = df[(df['state'] == 'COMPLETE') & df['value'].notnull()]
    param_cols = [c for c in df.columns if c not in ('number', 'value', 'state')]

    dfm = df.rename(columns={'number': 'trial'})
    dfm['value_mean'] = df.groupby(param_cols)['value'].transform('mean')
    dfm['trial_cnt'] = df.groupby(param_cols)['value'].transform('count')
    # Best params first: highest score when maximizing (stronger play),
    # lowest when minimizing (deliberately weaker play).
    best_first = direction != 'minimize'
    dff = dfm.sort_values(by=['value_mean', 'trial_cnt'],
                          ascending=[not best_first, False]).reset_index(drop=True)

    if elo_objective:
        dff['value_mean'] = dff['value_mean'].round(1)

    logger.info('Objective value mean and trial counts:')
    logger.info(dff.to_string(index=False))

    # Show the best param, value and trial number.
    logger.info(f'study best param: {study.best_params}')
    if elo_objective:
        logger.info(f'study best objective value: Elo {study.best_value}')
    else:
        logger.info(f'study best objective value: {study.best_value}')
    logger.info(f'study best trial number: {study.best_trial.number}\n')

    # Output for match manager.
    logger.info(f'{options_str(study.best_params)}\n')


def elapse_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    logger.info(f'elapse: {h:.0f}h:{m:.0f}m:{s:.0f}s')


def load_params_from_file(path):
    """
    Load a parameter dict from a JSON (.json) or YAML (.yaml/.yml) file,
    chosen by the file extension.
    """
    p = Path(path)
    if not p.is_file():
        raise Exception(f'Param file not found: {path}')

    # utf-8-sig transparently strips a leading BOM (common on Windows editors)
    # and still reads plain UTF-8 correctly.
    text = p.read_text(encoding='utf-8-sig')
    suffix = p.suffix.lower()

    if suffix == '.json':
        import json
        return json.loads(text)
    if suffix in ('.yaml', '.yml'):
        import yaml  # PyYAML, already a dependency of optuna
        return yaml.safe_load(text)

    raise Exception(
        f'Unsupported param file type "{suffix}" for {path}; use .json, .yaml or .yml.')


def resolve_param(inline_value, file_path, required, name, config_value=None):
    """
    Resolve a parameter dict. Precedence: --<name>-file, then inline --<name>,
    then the matching section of the unified --config file. Inline strings are
    parsed as Python literals (single-quoted dicts) for backward compatibility.
    """
    if file_path is not None:
        return load_params_from_file(file_path)
    if inline_value is not None:
        return ast.literal_eval(inline_value)
    if config_value is not None:
        return config_value
    if required:
        raise Exception(f'Define {name} via --{name}, --{name}-file, or the config file.')
    return None


def validate_input_param(input_param):
    """
    Check every param spec has default/min/max/step with sane values, so a
    config typo fails here with a clear message instead of a KeyError later.
    """
    for name, spec in input_param.items():
        if not isinstance(spec, dict):
            raise Exception(
                f'input param "{name}" must be a mapping with default, min, max and step.')
        missing = [k for k in ('default', 'min', 'max', 'step') if k not in spec]
        if missing:
            raise Exception(f'input param "{name}" is missing: {", ".join(missing)}.')
        if spec.get('type', 'int') not in ('int', 'float'):
            raise Exception(
                f'input param "{name}" has unsupported type "{spec["type"]}"; use int or float.')
        if spec['step'] <= 0:
            raise Exception(f'input param "{name}" step must be positive.')
        if spec['min'] > spec['max']:
            raise Exception(f'input param "{name}" min {spec["min"]} exceeds max {spec["max"]}.')
        if not spec['min'] <= spec['default'] <= spec['max']:
            raise Exception(
                f'input param "{name}" default {spec["default"]} is outside '
                f'[{spec["min"]}, {spec["max"]}].')


def detect_opening_format(opening_file):
    """
    Derive the cutechess/fastchess opening 'format' token from the opening file
    extension: .pgn -> 'pgn'; .epd/.fen -> 'epd' (both managers consume the EPD
    form). Any other extension stops the run with an error logged to both the
    console and log_tuner.txt.
    """
    suffix = Path(opening_file).suffix.lower()
    if suffix == '.pgn':
        return 'pgn'
    if suffix in ('.epd', '.fen'):
        return 'epd'
    msg = (f'Unsupported opening file extension "{suffix}" for "{opening_file}"; '
           f'use .pgn, .epd or .fen.')
    logger.error(msg)
    raise SystemExit(1)


def mapping_to_cli_tokens(mapping):
    """
    Convert a config mapping such as {'name': 'tpe', 'multivariate': True} into
    the list-of-lists token form that get_sampler()/get_pruner() expect, e.g.
    [['name=tpe', 'multivariate=true']]. Booleans are lowercased so the existing
    "== 'true'" checks keep working.
    """
    tokens = []
    for k, v in mapping.items():
        if isinstance(v, bool):
            v = 'true' if v else 'false'
        tokens.append(f'{k}={v}')
    return [tokens]


def options_to_defaults(cfg_options, parser):
    """
    Translate the 'options' section of a --config file into argparse defaults.

    Returns (defaults, sampler_tokens, pruner_tokens, hyperband_tokens).
    'defaults' is a dict for parser.set_defaults(); sampler/threshold_pruner/
    hyperband_pruner are returned separately because they use a list-of-lists
    token structure rather than a plain value.
    Keys are normalized ('-' -> '_') and validated against the known flags so
    typos are caught early.
    """
    valid_dests = {a.dest for a in parser._actions
                   if a.dest not in ('help', 'version', 'config')}

    defaults = {}
    sampler_tokens = None
    pruner_tokens = None
    hyperband_tokens = None

    for raw_key, value in cfg_options.items():
        key = raw_key.replace('-', '_')
        if key not in valid_dests:
            raise Exception(
                f'Unknown option "{raw_key}" in the config "options" section. '
                f'Valid options: {", ".join(sorted(valid_dests))}.')
        if key == 'sampler':
            sampler_tokens = mapping_to_cli_tokens(value) if isinstance(value, dict) else value
        elif key == 'threshold_pruner':
            pruner_tokens = mapping_to_cli_tokens(value) if isinstance(value, dict) else value
        elif key == 'hyperband_pruner':
            hyperband_tokens = mapping_to_cli_tokens(value) if isinstance(value, dict) else value
        else:
            defaults[key] = value

    return defaults, sampler_tokens, pruner_tokens, hyperband_tokens


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog='%s %s' % (__script_name__, __version__),
        description='Optimize parameter values of a game agent using optuna framework.',
        epilog='%(prog)s')
    parser.add_argument('--engine', required=False,
                        help='Engine filename or engine path and filename.\n'
                             'Required, but may instead be set via the --config file '
                             '(options.engine).')
    parser.add_argument('--draw-movenumber', required=False,
                        help='Number of moves reached before applying the draw adjudication.\n'
                             'If not specified then draw adjudication will be disabled.\n'
                             'This should be used together with --draw-movecount and --draw-score. Example:\n'
                             '--draw-movenumber 40 --draw-movecount 6 --draw-score 0.')
    parser.add_argument('--draw-movecount', required=False,
                        help='Number of move count reached before applying the draw adjudication, default=6.\n'
                             'This should be used together with --draw-movenumber and --draw-score. Example:\n'
                             '--draw-movenumber 40 --draw-movecount 6 --draw-score 0.',
                        default=6)
    parser.add_argument('--draw-score', required=False,
                        help='Score is in cp, default=0.\n'
                             'This should be used together with --draw-movenumber and --draw-movecount. Example:\n'
                             '--draw-movenumber 40 --draw-movecount 6 --draw-score 0.',
                        default=0)
    parser.add_argument('--resign-movecount', required=False,
                        help='Number of move counts before the game is adjudicated as a loss.\n'
                             'This should be used together with --resign-score option. Example:\n'
                             '--resign-movecount 10 --resign-score 700\n'
                             'Will terminate the game when there are 10 successive -700 or worse score.')
    parser.add_argument('--resign-score', required=False,
                        help='Score is centipawn where the game is considered resignable.\n'
                             'This should be used together with --resign-movecount option.')
    parser.add_argument('--trials', required=False, type=int,
                        help='Trials to try, default=1000.',
                        default=1000)
    parser.add_argument('--direction', required=False, type=str,
                        help='Objective direction can be minimize or maximize, default=maximize.',
                        default='maximize')
    parser.add_argument('--concurrency', required=False, type=int,
                        help='Number of game matches to run concurrently, default=1.',
                        default=1)
    parser.add_argument('--use-affinity', required=False, type=str,
                        nargs='?', const=True, default=None,
                        help='fastchess only: bind engine threads to CPU cores.\n'
                             'Bare --use-affinity auto-binds; or pass a core list/range,\n'
                             'e.g. --use-affinity 3,5,7-11,13. Reduces result variance.')
    parser.add_argument('--games-per-trial', required=False, type=int,
                        help='Number of games per trial, default=32.\n'
                        'This should be even number.', default=32)
    parser.add_argument('--study-name', required=False, type=str,
                        default='default_study_name',
                        help='The name of study. This can be used to resume\n'
                             'study sessions, default=default_study_name.')
    parser.add_argument('--base-time-sec', required=False, type=int,
                        help='Base time in sec for time control, default=5.',
                        default=5)
    parser.add_argument('--inc-time-sec', required=False, type=float,
                        help='Increment time in sec for time control, default=0.05.',
                        default=0.05)
    parser.add_argument('--depth', required=False, type=int,
                        help='The maximum depth that the engine is allowed to search.')
    parser.add_argument('--nodes', required=False, type=int,
                        help='The maximum nodes that the engine is allowed to search.\n'
                             'Supported by the pymatch and cutechess match managers\n'
                             '(pymatch with cecp engines needs the engine to have "feature nps").')
    parser.add_argument('--opening-file', required=False, type=str,
                        help='Start opening filename in pgn, fen or epd format.\n'
                             'The format is auto-detected from the extension\n'
                             '(.pgn -> pgn; .epd/.fen -> epd); other extensions stop the run.\n'
                             'Required, but may instead be set via the --config file '
                             '(options.opening_file).')
    parser.add_argument('--opening-format', required=False, type=str,
                        help='Deprecated/ignored: the format is auto-detected from the\n'
                             'opening file extension (.pgn -> pgn; .epd/.fen -> epd).',
                        default=None)
    parser.add_argument('--opening-posperfile', required=False, type=int,
                        help='number of positions per startpos file for duel.py match manager only.\n'
                        'If not specified then all positions will be considered.',
                        default=-1)
    parser.add_argument('--variant', required=False, type=str,
                        help='Game variant, default=normal.', default='normal')
    parser.add_argument('--pgn-output', required=False, type=str,
                        help='Output pgn filename, default=optuna_games.pgn.',
                        default='optuna_games.pgn')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=False,
                        help='Output plots in png. Use --no-plot to override a config "plot: true".')
    parser.add_argument('--elo-objective', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable the elo as objective value instead of the default score rate.\n'
                             'Use --no-elo-objective to override a config value.')
    parser.add_argument('--noisy-result', action=argparse.BooleanOptionalAction, default=False,
                        help='A flag to replay engine vs engine match when sampler repeats suggesting same parameter values.\n'
                             'When you play an engine vs engine match at fixed depth, generally the result is not noisy.\n'
                             'When the sampler suggests a param that was already suggested before we just return the result from\n'
                             'the previous trial. However if you play an engine vs engine match with TC, the result is noisy\n'
                             'specially when number of games is low. If this flag is enabled, we replay an engine match even if\n'
                             'the param was already suggested before.')
    parser.add_argument('--save-plots-every-trial', required=False, type=int,
                        help='Save plots every n trials, default=10.',
                        default=10)
    parser.add_argument('--match-manager', required=False, type=str,
                        help='The application that handles the engine match, can be\n'
                             'pymatch, cutechess, fastchess or duel, default=pymatch.\n'
                             'pymatch is built-in (uci and cecp engines, standard chess\n'
                             'only, needs the python-chess package).\n'
                             'Note: fastchess is UCI-only (no xboard/cecp engines).',
                        default=None)
    parser.add_argument('--match-manager-file', required=False, type=str,
                        default=None,
                        help='Path to the match-manager file. For cutechess/fastchess\n'
                             'this is the executable (e.g. a self-compiled one on\n'
                             'Linux); it overrides the bundled binary and PATH lookup.\n'
                             'For duel it is the duel.py script. Pair it with the\n'
                             'matching --match-manager flavor. Examples:\n'
                             '--match-manager cutechess --match-manager-file /home/user/cutechess/cutechess-cli\n'
                             '--match-manager fastchess --match-manager-file /home/user/fastchess/fastchess\n'
                             '--match-manager duel --match-manager-file /home/user/duel.py')
    parser.add_argument('--protocol', required=False, type=str,
                        help='The protocol that the engine supports, can be'
                             ' uci or cecp, default=uci.',
                        default='uci')
    parser.add_argument('--sampler', required=False, nargs='*', action='append',
                        metavar=('name=', 'option_name='),
                        help='The sampler to be used in the study, default name=tpe.\n'
                             'name can be tpe or cmaes, examples:\n'
                             '--sampler name=tpe n_ei_candidates=50 multivariate=true group=true seed=100 constant_liar=true n_startup_trials=6 ...\n'
                             '  default values: n_ei_candidates=24, multivariate=false, group=false, seed=None, constant_liar=false, n_startup_trials=10\n'
                             '  TPE ref: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html\n'
                             '--sampler name=cmaes sigma0=20 n_startup_trials=6 seed=100 ...\n'
                             '  default values: sigma0 or initial std deviation is None, n_startup_trials=1, seed=None.\n'
                             '  This tells cmaes that optimal parameter values\n'
                             '  lies within init_value +/- 3 * sigma0. By default this value is the parameter minimum_range/6.\n'
                             '  ref: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html\n')
    parser.add_argument('--threshold-pruner', required=False, nargs='*', action='append',
                        metavar=('result=', 'games='),
                        help='A trial pruner used to prune or stop unpromising trials. Example:\n'
                             'tuner.py --games-per-trial 100 --threshold-pruner result=0.45 games=50 interval=1 ...\n'
                             'After 50 partial games, check the score of the match, if this is below 0.45, then prune\n'
                             'the trial or stop the engine match, get new param from optimizer and start a new trial.\n'
                             'Default values: result=0.25, games=games_per_trial/2, interval=1\n'
                             'If --elo-objective flag is enabled, the result is in Elo, example:\n'
                             'tuner.py --games-per-trial 100 --elo-objective --threshold-pruner result=-10 ...\n'
                             'That would mean after 50 partial games, when result is below -10 Elo\n'
                             'then the trial is pruned, the other 50 games will not be played.',
                        default=None)
    parser.add_argument('--hyperband-pruner', required=False, nargs='*', action='append',
                        metavar=('min_resource=', 'reduction_factor='),
                        help='A Hyperband trial pruner that early-stops unpromising trials.\n'
                             'Mutually exclusive with --threshold-pruner. The resource unit is\n'
                             'games played: the match is reported every min_resource games and\n'
                             'Hyperband prunes weak trials at its rungs. Example:\n'
                             'tuner.py --games-per-trial 1000 --hyperband-pruner min_resource=50 reduction_factor=3 ...\n'
                             'Options: min_resource, max_resource, reduction_factor, bootstrap_count, interval.\n'
                             'Default values: min_resource=games_per_trial/4 (even), max_resource=games_per_trial,\n'
                             'reduction_factor=3, bootstrap_count=0, interval=1.\n'
                             'Ref: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html',
                        default=None)
    parser.add_argument('--input-param', required=False, type=str,
                        help='The parameters that will be optimized. Define this OR\n'
                             '--input-param-file (one is required).\n'
                             'Example 1 with 1 parameter:\n'
                             '--input-param \"{\'pawn\': {\'default\': 92,'
                             ' \'min\': 90, \'max\': 120, \'step\': 2}}\"\n'
                             'Example 2 with 2 parameters:\n'
                             '--input-param \"{\'pawn\': {\'default\': 92,'
                             ' \'min\': 90, \'max\': 120, \'step\': 2},'
                             ' \'knight\': {\'default\': 300, \'min\': 250,'
                             ' \'max\': 350, \'step\': 2}}\"\n'
                             'Example 3 with 1 parameter but float value:\n'
                             '--input-param \"{\'CPuct\': {\'default\': 0.5,'
                             ' \'min\': 0.1, \'max\': 3.0, \'step\': 0.05, \'type\': \'float\'}}\"'
                        )
    parser.add_argument('--input-param-file', required=False, type=str,
                        help='Path to a JSON (.json) or YAML (.yaml/.yml) file with the\n'
                             'parameters to optimize, as an alternative to --input-param.\n'
                             'Same structure, e.g. a JSON file containing:\n'
                             '{"pawn": {"default": 92, "min": 90, "max": 120, "step": 2}}')
    parser.add_argument('-v', '--version', action='version', version=f'{__version__}')
    parser.add_argument('--common-param', required=False, type=str,
                        help='The parameters that will be sent to both test and base engines.\n'
                             'Make sure that this param is not included in the input-param.\n'
                             'Example:\n'
                             '--common-param \"{\'RookOpenFile\': 92, \'KnightOutpost\': 300}\"')
    parser.add_argument('--common-param-file', required=False, type=str,
                        help='Path to a JSON or YAML file with the common parameters,\n'
                             'as an alternative to --common-param.')
    parser.add_argument('--config', required=False, type=str,
                        help='Path to a single unified YAML/JSON config file holding\n'
                             'input_param, common_param and an options section (the\n'
                             'other flags, keyed by their long names with underscores).\n'
                             'Any flag given on the command line overrides the config.\n'
                             'Example: --config yaml_files/deuterium_config.yaml')

    # First pass: read --config only, then fold its "options" section in as
    # argparse defaults so that values still on the command line take precedence.
    # --engine/--opening-file are optional here and validated after the merge.
    pre_args, _ = parser.parse_known_args()

    cfg = {}
    cfg_sampler_tokens = None
    cfg_pruner_tokens = None
    cfg_hyperband_tokens = None
    if pre_args.config is not None:
        cfg = load_params_from_file(pre_args.config)
        if not isinstance(cfg, dict):
            raise Exception(
                f'Config file {pre_args.config} must be a mapping with '
                f'input_param / common_param / options sections.')
        cfg_options = cfg.get('options') or {}
        defaults, cfg_sampler_tokens, cfg_pruner_tokens, cfg_hyperband_tokens = options_to_defaults(cfg_options, parser)
        parser.set_defaults(**defaults)

    args = parser.parse_args()

    elo_objective = args.elo_objective

    # Engine and opening file are required, from either the CLI or the config.
    if args.engine is None:
        raise Exception('Define the engine via --engine or the config "options.engine".')
    if args.opening_file is None:
        raise Exception(
            'Define the opening file via --opening-file or the config "options.opening_file".')

    # Auto-detect the opening format from the file extension (authoritative).
    args.opening_format = detect_opening_format(args.opening_file)
    logger.info(f'opening file: {args.opening_file}, format: {args.opening_format}')

    # Check if engine and opening files exist.
    eng_path = Path(args.engine)
    if not eng_path.is_file():
        raise Exception(f'The engine in {eng_path} is missing!')
    if not Path(args.opening_file).is_file():
        raise Exception(f'The opening file in {args.opening_file} is missing!')

    # depth and nodes are both search limits; with both given, one would
    # silently win, so make the choice explicit.
    if args.depth is not None and args.nodes is not None:
        raise Exception('Define only one of --depth and --nodes.')

    # Resolve and validate the match manager.
    # --match-manager defaults to None so we can tell "explicitly set" from "omitted".
    valid_managers = ('pymatch', 'cutechess', 'fastchess', 'duel')
    if args.match_manager is None:
        if args.match_manager_file is not None:
            # A path was given but the flavor was not; the command syntax differs
            # between managers, so we cannot safely guess it.
            raise Exception(
                '--match-manager-file was given without --match-manager. Please also '
                'specify the flavor, e.g. --match-manager fastchess (or cutechess).')
        args.match_manager = 'pymatch'
    if args.match_manager not in valid_managers:
        raise Exception(
            f'Unknown --match-manager "{args.match_manager}", '
            f'expected one of {", ".join(valid_managers)}.')

    # The built-in pymatch manager: in-process, standard chess only, needs
    # the python-chess package.
    if args.match_manager == 'pymatch':
        if args.match_manager_file is not None:
            raise Exception(
                '--match-manager-file does not apply to the built-in pymatch manager.')
        if args.variant != 'normal':
            raise Exception(
                'The built-in pymatch manager supports standard chess only; '
                'use --match-manager cutechess for variants.')
        try:
            import chess  # noqa: F401
        except ModuleNotFoundError:
            raise Exception(
                'The default pymatch match manager needs the python-chess package. '
                'Install it with "pip install chess" (see requirements.txt), or '
                'choose --match-manager cutechess.')

    # fastchess is UCI-only; it cannot run xboard/cecp engines.
    if args.match_manager == 'fastchess' and args.protocol == 'cecp':
        raise Exception(
            'fastchess only supports UCI engines; --protocol cecp is not compatible '
            'with --match-manager fastchess.')

    # --use-affinity is a fastchess-only feature.
    if args.use_affinity is not None and args.match_manager != 'fastchess':
        raise Exception(
            '--use-affinity is only supported with --match-manager fastchess.')

    # duel.py reads line-based books only and has no node limit.
    if args.match_manager == 'duel':
        if args.opening_format == 'pgn':
            raise Exception(
                'The duel match manager only supports epd/fen opening files, not pgn.')
        if args.nodes is not None:
            raise Exception(
                '--nodes is not supported by the duel match manager; use --depth.')

    # Check the user-supplied match-manager executable, if any, exists.
    if args.match_manager_file is not None and not Path(args.match_manager_file).is_file():
        raise Exception(
            f'The match-manager file in {args.match_manager_file} is missing!')

    trials = args.trials

    if elo_objective:
        init_value = 0.0
    else:
        init_value = 0.5

    save_plots_every_trial = args.save_plots_every_trial
    # A block of 0 trials would never advance the trial counter.
    if save_plots_every_trial < 1:
        raise Exception('--save-plots-every-trial must be 1 or higher.')

    # Common param can come from --common-param (inline) or --common-param-file.
    common_param = resolve_param(args.common_param, args.common_param_file,
                                 required=False, name='common-param',
                                 config_value=cfg.get('common_param'))

    # Number of games should be even for a fair engine match.
    games_per_trial = args.games_per_trial
    games_per_trial += 1 if (args.games_per_trial % 2) != 0 else 0

    study_name = args.study_name
    storage_file = f'{study_name}.db'

    logger.info(f'{__script_name__} {__version__}')
    logger.info(f'trials: {trials}, games_per_trial: {games_per_trial}')
    logger.info(f'sampler: {args.sampler}')
    logger.info(f'objective value type: {"Elo" if elo_objective else "score rate"}\n')

    # Input param can come from --input-param (inline) or --input-param-file.
    input_param = resolve_param(args.input_param, args.input_param_file,
                                required=True, name='input-param',
                                config_value=cfg.get('input_param'))
    input_param = OrderedDict(sorted(input_param.items()))
    validate_input_param(input_param)

    # A param in both would reach the engine twice and the common (fixed)
    # value would silently override the tuned one.
    if common_param is not None:
        overlap = sorted(set(input_param) & set(common_param))
        if overlap:
            raise Exception(
                f'These params are in both input_param and common_param: '
                f'{", ".join(overlap)}. Remove them from one of the two.')

    logger.info(f'input param: {input_param}\n')
    init_param = Objective.set_param(input_param)

    # Adjust save_plots_every_trial if trials is lower than it so a block is
    # at least 1 trial. The plot will be generated after each block.
    if trials < save_plots_every_trial:
        save_plots_every_trial = trials

    # Run in blocks of save_plots_every_trial so plots/logs are saved after
    # each block. The last block runs whatever is left, so a --trials count
    # that is not a multiple of save_plots_every_trial is honoured exactly
    # instead of being truncated.
    trials_done = 0

    # Define sampler to use, default is TPE. CLI --sampler wins; otherwise fall
    # back to the config options.sampler mapping (already tokenized).
    sampler_arg = args.sampler if args.sampler is not None else cfg_sampler_tokens
    sampler, n_startup_trials = Objective.get_sampler(sampler_arg)

    # ThresholdPruner as trial pruner, if result of a match is below result
    # threshold after games threshold then prune the trial. Get new param
    # from optimizer and continue with the next trial.
    # --threshold-pruner result=0.45 games=50 --games-per-trial 100 ...
    threshold_arg = args.threshold_pruner if args.threshold_pruner is not None else cfg_pruner_tokens
    hyperband_arg = args.hyperband_pruner if args.hyperband_pruner is not None else cfg_hyperband_tokens
    if threshold_arg is not None and hyperband_arg is not None:
        raise SystemExit('Define only one pruner: --threshold-pruner or --hyperband-pruner '
                         '(or config options.threshold_pruner / options.hyperband_pruner), not both.')
    # Hyperband compares intermediate values through the study direction, so it
    # needs nothing from us. The threshold pruner has a fixed bound and does,
    # hence the direction is passed to get_pruner().
    pruner, th_pruner = Objective.get_pruner(threshold_arg, hyperband_arg,
                                             games_per_trial, elo_objective,
                                             args.direction)

    # Create (or resume) the study once; the loop below only runs more trials
    # on it. Recreating it per block would rescan all trials every time.
    study = optuna.create_study(study_name=study_name,
                                direction=args.direction,
                                storage=f'sqlite:///{storage_file}',
                                load_if_exists=True, sampler=sampler,
                                pruner=pruner)

    # A resumed study keeps the direction it was created with, and optuna
    # ignores the one asked for here without saying so. Fail loudly instead:
    # silently strengthening a study meant to be weakened wastes a whole run.
    if study.direction.name.lower() != args.direction.lower():
        raise SystemExit(
            f'Study "{study_name}" already exists with direction '
            f'{study.direction.name.lower()}, but --direction {args.direction} was given. '
            f'Use --direction {study.direction.name.lower()} to continue it, '
            f'or pick a new --study-name.')

    # Report the best trial of a previous study session, if any.
    is_study = False
    try:
        best = study.best_trial
        is_study = True
        logger.info(f'study best objective value: {best.value}')
        logger.info(f'study best param: {best.params}')
    except ValueError:
        logger.warning('Warning, best value/param from a previous session is not found!')
    except Exception:
        logger.exception('Unexpected error while reading study best value/param.')
        raise

    # If there is no trial recorded yet we will initialize our study
    # with default values from the engine.
    if not is_study:
        distri = {}
        for k, v in input_param.items():
            if v.get('type') == 'float':
                distri[k] = FloatDistribution(v['min'], v['max'], log=False, step=v['step'])
            else:
                distri[k] = IntDistribution(v['min'], v['max'], log=False, step=v['step'])

        init_trial = optuna.trial.create_trial(
            params=copy.deepcopy(init_param),
            distributions=distri,
            value=init_value,
        )
        study.add_trial(init_trial)

        logger.info(f"added trial: yes, params: {init_trial.params}, objective value: {init_trial.values[0]}")

    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    objective = Objective(
        study=study, engine=args.engine, input_param=input_param,
        init_param=init_param, init_value=init_value,
        variant=args.variant, opening_file=args.opening_file,
        opening_format=args.opening_format, pgnout=args.pgn_output,
        nodes=args.nodes, base_time_sec=args.base_time_sec,
        inc_time_sec=args.inc_time_sec, concurrency=args.concurrency,
        proto=args.protocol, match_manager=args.match_manager,
        match_manager_file=args.match_manager_file, depth=args.depth,
        games_per_trial=games_per_trial, threshold_pruner=th_pruner,
        common_param=common_param, resign_movecount=args.resign_movecount,
        resign_score=args.resign_score, draw_movenumber=args.draw_movenumber,
        draw_movecount=args.draw_movecount, draw_score=args.draw_score,
        opening_posperfile=args.opening_posperfile,
        n_startup_trials=n_startup_trials, noisy_result=args.noisy_result,
        elo_objective=elo_objective, use_affinity=args.use_affinity)

    logger.info('Starting optimization ...')

    while trials_done < trials:
        n_trials = min(save_plots_every_trial, trials - trials_done)
        trials_done += n_trials

        study.optimize(objective, n_trials=n_trials)

        # Create and save plots after each block of trials.
        save_plots(study, study_name, input_param, args.plot)

        # Build pandas dataframe, print and save to csv file.
        save_study_log(study, study_name, elo_objective, args.direction)


if __name__ == "__main__":
    main()
