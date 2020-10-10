"""
duel.py

A module to handle xboard or winboard engine matches.
"""


import subprocess
import argparse
import time
import random
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import logging
from statistics import mean
from typing import List


logging.basicConfig(
    filename='log_duel.txt', filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - pid%(process)5d - %(levelname)5s - %(message)s')


class Timer:
    def __init__(self, base_time, inc_time):
        """
        The time unit is in ms (milliseconds)
        """
        self.base_time = base_time
        self.inc_time = inc_time
        self.rem_time = self.base_time + self.inc_time

    def update(self, elapse):
        """
        This is called after every engine move is completed.
        """
        self.rem_time -= elapse
        self.rem_time += self.inc_time

    def is_zero_time(self):
        return True if self.rem_cs() <= 0 else False

    def rem_cs(self):
        return self.rem_time // 10


def define_engine(engine_option_value):
    """
    Define engine files, name and options.
    """
    ed1, ed2 = {}, {}
    e1 = {'proc': None, 'cmd': None, 'name': 'test', 'opt': ed1, 'tc': '', 'depth': 0}
    e2 = {'proc': None, 'cmd': None, 'name': 'base', 'opt': ed2, 'tc': '', 'depth': 0}
    for i, eng_opt_val in enumerate(engine_option_value):
        for value in eng_opt_val:
            if i == 0:
                if 'cmd=' in value:
                    e1.update({'cmd': value.split('=')[1]})
                elif 'option.' in value:
                    # Todo: support float value
                    # option.QueenValueOpening=1000
                    optn = value.split('option.')[1].split('=')[0]
                    optv = int(value.split('option.')[1].split('=')[1])
                    ed1.update({optn: optv})
                    e1.update({'opt': ed1})
                elif 'tc=' in value:
                    e1.update({'tc': value.split('=')[1]})
                elif 'name=' in value:
                    e1.update({'name': value.split('=')[1]})
                elif 'depth=' in value:
                    e1.update({'depth': int(value.split('=')[1])})
            elif i == 1:
                if 'cmd=' in value:
                    e2.update({'cmd': value.split('=')[1]})
                elif 'option.' in value:
                    optn = value.split('option.')[1].split('=')[0]
                    optv = int(value.split('option.')[1].split('=')[1])
                    ed2.update({optn: optv})
                    e2.update({'opt': ed2})
                elif 'tc=' in value:
                    e2.update({'tc': value.split('=')[1]})
                elif 'name=' in value:
                    e2.update({'name': value.split('=')[1]})
                elif 'depth=' in value:
                    e2.update({'depth': int(value.split('=')[1])})

    return e1, e2


def get_fen_list(fn, is_rand=False):
    """
    Read fen file and return a list of fens.
    """
    fens = []

    if fn is None:
        return fens

    with open(fn) as f:
        for lines in f:
            fen = lines.strip()
            fens.append(fen)

    if is_rand:
        random.shuffle(fens)

    return fens


def get_tc(tcd):
    """
    tc=0/3+1 or 3+1, blitz 3m + 1s inc
    tc=0/0:5+0.1 or 0:5+0.1, blitz 0m + 5s + 0.1s inc
    """
    base_minv, base_secv, inc_secv = 0, 0, 0.0

    if tcd == '':
        return base_minv, base_secv, inc_secv

    # Check base time with minv:secv format.
    if '/' in tcd:
        basev = tcd.split('/')[1].split('+')[0].strip()
    else:
        basev = tcd.split('+')[0].strip()

    if ':' in basev:
        base_minv = int(basev.split(':')[0])
        base_secv = int(basev.split(':')[1])
    else:
        base_minv = int(basev)

    if '/' in tcd:
        inc_secv = float(tcd.split('/')[1].split('+')[1].strip())
    else:
        inc_secv = float(tcd.split('+')[1].strip())

    return base_minv, base_secv, inc_secv


def turn(fen):
    """
    Return side to move of the given fen.
    """
    side = fen.split()[1].strip()
    if side == 'w':
        return True
    return False


def save_game(outfn, fen, moves, scores, depths, e1, e2, start_turn, gres,
              termination='', variant=''):
    logging.info('Saving game ...')
    with open(outfn, 'a') as f:
        f.write('[Event "Optimization test"]\n')
        f.write(f'[White "{e1 if start_turn else e2}"]\n')
        f.write(f'[Black "{e1 if not start_turn else e2}"]\n')
        f.write(f'[Result "{gres}"]\n')

        f.write(f'[Variant "{variant}"]\n')

        if termination != '':
            f.write(f'[Termination "{termination}"]\n')

        if not isinstance(fen, int):
            f.write(f'[FEN "{fen}"]\n\n')
        else:
            f.write('\n')

        for i, (m, s, d) in enumerate(zip(moves, scores, depths)):
            num = i + 1
            if num % 2 == 0:
                if start_turn:
                    str_num = f'{num // 2}... '
                else:
                    str_num = f'{num // 2}. '
            else:
                num += 1
                if start_turn:
                    str_num = f'{num // 2}. '
                else:
                    str_num = f'{num // 2}... '
            f.write(f'{str_num}{m} {{{s}/{d}}} ')
            if (i + 1) % 5 == 0:
                f.write('\n')
        f.write('\n\n')


def adjudicate_win(score_history, resign_option, side):
    logging.info('Try adjudicating this game by win ...')
    ret, gres, e1score = False, '*', 0.0

    if len(score_history) >= 40:
        fcp_score = score_history[0::2]
        scp_score = score_history[1::2]

        fwin_cnt, swin_cnt = 0, 0
        movecount = resign_option['movecount'] * 2
        score = resign_option['score']

        for i, (fs, ss) in enumerate(zip(reversed(fcp_score),
                                         reversed(scp_score))):
            if i >= movecount:
                break
            if i <= movecount and fs >= score and ss <= -score:
                fwin_cnt += 1
            elif i <= movecount and fs <= -score and ss >= score:
                swin_cnt += 1

        if fwin_cnt >= movecount:
            gres = '1-0' if side else '0-1'
            e1score = 1.0
            logging.info(f'{"White" if side else "Black"} wins by adjudication.')
            ret = True
        if swin_cnt >= movecount:
            gres = '1-0' if side else '0-1'
            e1score = 0
            logging.info(f'{"White" if side else "Black"} wins by adjudication.')
            ret = True

    return ret, gres, e1score


def adjudicate_draw(score_history, draw_option):
    logging.info('Try adjudicating this game by draw ...')
    ret, gres, e1score = False, '*', 0.0

    if len(score_history) >= draw_option['movenumber'] * 2:
        fcp_score = score_history[0::2]
        scp_score = score_history[1::2]

        draw_cnt = 0
        movecount = draw_option['movecount'] * 2
        score = draw_option['score']

        for i, (fs, ss) in enumerate(zip(reversed(fcp_score),
                                         reversed(scp_score))):
            if i >= movecount:
                break
            if (i <= movecount and abs(fs) <= score
                    and abs(ss) <= score):
                draw_cnt += 1

        if draw_cnt >= movecount:
            gres = '1/2-1/2'
            e1score = 0.5
            logging.info('Draw by adjudication.')
            ret = True

    return ret, gres, e1score


def is_game_end(line, test_engine_color):
    game_end, gres, e1score, termination, comment = False, '*', 0.0, '', ''

    if '1-0' in line:
        game_end = True
        e1score = 1.0 if test_engine_color else 0.0
        gres = '1-0'
        termination = 'white mates black'
    elif '0-1' in line:
        game_end = True
        e1score = 1.0 if not test_engine_color else 0.0
        gres = '0-1'
        termination = 'black mates white'
    elif '1/2-1/2' in line:
        game_end = True
        e1score = 0.5
        gres = '1/2-1/2'
        if 'repetition' in line.lower():
            termination = 'draw by repetition'
        elif 'insufficient' in line.lower():
            termination = 'draw by insufficient mating material'
        elif 'fifty' in line.lower():
            termination = 'draw by insufficient mating material'
        elif 'stalemate' in line.lower():
            termination = 'draw by stalemate'

    return game_end, gres, e1score, termination


def param_to_dict(param):
    """
    Convert string param to a dictionary.
    """
    ret_param = {}
    for par in param.split(','):
        par = par.strip()
        sppar = par.split()  # Does not support param with space
        spname = sppar[0].strip()
        spvalue = int(sppar[1].strip())
        ret_param.update({spname: spvalue})

    return ret_param


def time_forfeit(is_timeup, current_color, test_engine_color):
    game_end, gres, e1score = False, '*', 0.0

    if is_timeup:
        # test engine loses as white
        if current_color and test_engine_color:
            gres = '0-1'
            e1score = 0.0
            game_end = True
            print(f'test engine with color {test_engine_color} loses on time')
        # test engine loses as black
        elif not current_color and not test_engine_color:
            gres = '1-0'
            e1score = 0.0
            game_end = True
            print(f'test engine with color {test_engine_color} loses on time')
        # test engine wins as white
        elif not current_color and test_engine_color:
            gres = '1-0'
            e1score = 1.0
            game_end = True
            print(f'test engine with color {test_engine_color} wins on time')
        # test engine wins as black
        elif current_color and not test_engine_color:
            gres = '0-1'
            e1score = 1.0
            game_end = True
            print(f'test engine with color {test_engine_color} wins on time')

    if game_end:
        logging.info('Game ends by time forfeit.')

    return game_end, gres, e1score


def match(e1, e2, fen, output_game_file, variant, draw_option,
          resign_option, repeat=2) -> List[float]:
    """
    Run an engine match between e1 and e2. Save the game and print result
    from e1 perspective.
    """
    move_hist = []
    all_e1score = []
    is_show_search_info = False

    # Start engine match, 2 games will be played.
    for gn in range(repeat):
        logging.info(f'Match game no. {gn + 1}')
        logging.info(f'Test engine plays as {"first" if gn % 2 == 0 else "second"} engine.')

        pe1 = subprocess.Popen(e1['cmd'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)

        pe2 = subprocess.Popen(e2['cmd'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)

        e1.update({'proc': pe1})
        e2.update({'proc': pe2})

        if gn % 2 == 0:
            eng = [e1, e2]
        else:
            eng = [e2, e1]

        for i, pr in enumerate(eng):
            e = pr['proc']
            pn = pr['name']

            e.stdin.write('xboard\n')
            logging.debug(f'{pn} > xboard')
            e.stdin.write('protover 2\n')
            logging.debug(f'{pn} > protover 2')

            for eline in iter(e.stdout.readline, ''):
                line = eline.strip()
                logging.debug(f'{pn} < {line}')
                if 'done=1' in line:
                    break

            # Set param to engines.
            for k, v in pr['opt'].items():
                e.stdin.write(f'option {k}={v}\n')
                logging.debug(f'{pn} > option {k}={v}')

        timer, depth_control = [], []
        for i, pr in enumerate(eng):
            e = pr['proc']
            pn = pr['name']

            e.stdin.write(f'variant {variant}\n')
            logging.debug(f'{pn} > variant {variant}')

            e.stdin.write('ping 1\n')
            logging.debug(f'{pn} > ping 1')
            for eline in iter(e.stdout.readline, ''):
                line = eline.strip()
                logging.debug(f'{pn} < {line}')
                if 'pong' in line:
                    break

            e.stdin.write('new\n')
            logging.debug(f'{pn} > new')

            e.stdin.write('post\n')
            logging.debug(f'{pn} > post')

            # Define time control, base time in minutes and inc in seconds.
            base_minv, base_secv, incv = get_tc(pr['tc'])
            all_base_sec = base_minv * 60 + base_secv

            logging.info(f'base_minv: {base_minv}m, base_secv: {base_secv}s, incv: {incv}s')

            # Send level command to each engine.
            tbase = max(1, all_base_sec//60)
            e.stdin.write(f'level 0 {tbase} {float(incv):0.2f}\n')
            logging.debug(f'{pn} > level 0 {tbase} {float(incv):0.2f}')

            # Setup Timer, convert base time to ms and inc in sec to ms
            timer.append(Timer(all_base_sec * 1000, int(incv * 1000)))

            depth_control.append(pr['depth'])

            e.stdin.write('force\n')
            logging.debug(f'{pn} > force')

            e.stdin.write(f'setboard {fen}\n')
            logging.debug(f'{pn} > setboard {fen}')

            e.stdin.write('ping 2\n')
            logging.debug(f'{pn} > ping 2')
            for eline in iter(e.stdout.readline, ''):
                line = eline.strip()
                logging.debug(f'{pn} < {line}')
                if 'pong' in line:
                    break

        num, side, move, line, game_end = 0, 0, None, '', False
        score_history, elapse_history, depth_history = [], [], []
        start_turn = turn(fen) if not isinstance(fen, int) else True
        gres, e1score = '*', 0.0
        is_time_over = [False, False]
        current_color = start_turn  # True if white to move

        test_engine_color = True if ((start_turn and gn % 2 == 0) or (not start_turn and gn % 2 != 0)) else False
        termination = ''

        # Start the game.
        while True:
            if depth_control[side] > 0:
                eng[side]['proc'].stdin.write(f'sd {depth_control[side]}\n')
                logging.debug(f'{eng[side]["name"]} > sd {depth_control[side]}')
            else:
                eng[side]['proc'].stdin.write(f'time {timer[side].rem_cs()}\n')
                logging.debug(f'{eng[side]["name"]} > time {timer[side].rem_cs()}')

                eng[side]['proc'].stdin.write(f'otim {timer[not side].rem_cs()}\n')
                logging.debug(f'{eng[side]["name"]} > otim {timer[not side].rem_cs()}')

            t1 = time.perf_counter_ns()

            if num == 0:
                eng[side]['proc'].stdin.write('go\n')
                logging.debug(f'{eng[side]["name"]} > go')
            else:
                move_hist.append(move)
                eng[side]['proc'].stdin.write(f'{move}\n')
                logging.debug(f'{eng[side]["name"]} > {move}')

                # Send another go because of force.
                if num == 1:
                    eng[side]['proc'].stdin.write('go\n')
                    logging.debug(f'{eng[side]["name"]} > go')

            num += 1
            score, depth = None, None

            for eline in iter(eng[side]['proc'].stdout.readline, ''):
                line = eline.strip()

                logging.debug(f'{eng[side]["name"]} < {line}')

                if is_show_search_info:
                    if not line.startswith('#'):
                        print(line)

                # Save score and depth from engine search info.
                if line.split()[0].isdigit():
                    score = int(line.split()[1])  # cp
                    depth = int(line.split()[0])

                # Check end of game as claimed by engines.
                game_endr, gresr, e1scorer, termi = is_game_end(line, test_engine_color)
                if game_endr:
                    game_end, gres, e1score, termination = game_endr, gresr, e1scorer, termi
                    break

                if 'move ' in line and not line.startswith('#'):
                    elapse = (time.perf_counter_ns() - t1) // 1000000
                    timer[side].update(elapse)
                    elapse_history.append(elapse)

                    move = line.split('move ')[1]
                    score_history.append(score if score is not None else 0)
                    depth_history.append(depth if depth is not None else 0)

                    if timer[side].is_zero_time():
                        is_time_over[current_color] = True
                        termination = 'forfeits on time'
                        logging.info('time is over')
                    break

            if game_end:
                break

            # Game adjudications

            # Resign
            if (resign_option['movecount'] is not None
                    and resign_option['score'] is not None):
                game_endr, gresr, e1scorer = adjudicate_win(
                    score_history, resign_option, side)

                if game_endr:
                    gres, e1score = gresr, e1scorer
                    logging.info('Game ends by resign adjudication.')
                    break

            # Draw
            if (draw_option['movenumber'] is not None
                    and draw_option['movenumber'] is not None
                    and draw_option['score'] is not None):
                game_endr, gresr, e1scorer = adjudicate_draw(
                    score_history, draw_option)
                if game_endr:
                    gres, e1score = gresr, e1scorer
                    logging.info('Game ends by resign adjudication.')
                    break

            # Time is over
            if depth_control[side] == 0:
                game_endr, gresr, e1scorer = time_forfeit(
                    is_time_over[current_color], current_color, test_engine_color)
                if game_endr:
                    gres, e1score = gresr, e1scorer
                    break

            side = not side
            current_color = not current_color

        if output_game_file is not None:
            save_game(output_game_file, fen, move_hist, score_history,
                      depth_history, eng[0]["name"], eng[1]["name"],
                      start_turn, gres, termination, variant)

        for i, e in enumerate(eng):
            e['proc'].stdin.write('quit\n')
            logging.debug(f'{e["name"]} > quit')

        all_e1score.append(e1score)

    return all_e1score


def round_match(fen, e1, e2, output_game_file, repeat, draw_option,
                resign_option, variant, posround=1) -> List[float]:
    """
    Play a match between e1 and e2 using fen as starting position. By default
    2 games will be played color is reversed. If posround is more than 1, the
    match will be repeated posround times. The purpose of posround is to verify
    that the match result is repeatable with the use of only a single fen.
    """
    test_engine_score = []

    for _ in range(posround):
        res = match(e1, e2, fen, output_game_file, variant,
                    draw_option, resign_option, repeat=repeat)
        test_engine_score.append(res)

    return test_engine_score


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-rounds', required=False,
                        help='Number of encounter. Can be number of games if repeat is 2, default=2\n',
                        type=int, default=2)
    parser.add_argument('-repeat', required=False,
                        help='Number of times to play a certain opening\n'
                             'default is 2 so that the position will be\n'
                             'played twice and each engine takes both white\n'
                             'and black at the start of each game.',
                        type=int, default=2)
    parser.add_argument('-engine', nargs='*', action='append', required=True,
                        metavar=('cmd=', 'name='),
                        help='This option is used to define the engines.\n'
                        'Example:\n'
                        '-engine cmd=engine1.exe name=test ...'
                             ' --engine cmd=engine2.exe name=base')
    parser.add_argument('-draw', nargs='*', action='append', required=False,
                        metavar=('movenumber=', 'movecount='),
                        help='Adjudicates game to a draw result. Example:\n'
                             '-draw movenumber=40 movecount=10 score=0')
    parser.add_argument('-resign', nargs='*', action='append', required=False,
                        metavar=('movecount=', 'score='),
                        help='Adjudicates game to a loss result. Example:\n'
                             '-resign movecount=10 score=900')
    parser.add_argument('-pgnout', required=False,
                        metavar='pgn_output_filename',
                        help='pgn output filename')
    parser.add_argument('-concurrency', required=False,
                        help='number of game to run in parallel, default=1',
                        type=int, default=1)
    parser.add_argument('-variant', required=True, help='name of the variant')
    parser.add_argument('-each', nargs='*', action='append', required=False,
                        metavar=('tc=', 'option.<option_name>='),
                        help='This option is used to apply to both engnes.\n'
                             'Example where tc is applied to each engine:\n'
                             '-each tc=1+0.1')
    parser.add_argument('-openings', nargs='*', action='append',
                        required=False,
                        metavar=('file=', 'format='),
                        help='Define start openings. Example:\n'
                             '-openings file=start.fen format=epd')
    parser.add_argument('-tournament', required=False, default='round-robin',
                        metavar='tour_type',
                        help='tournament type, default=round-robin')

    args = parser.parse_args()

    # Define engine files, name and options.
    e1, e2 = define_engine(args.engine)

    # Exit if engine file is not defined.
    if e1['cmd'] is None or e2['cmd'] is None:
        print('Error, engines are not properly defined!')
        return

    each_engine_option = {}
    if args.each is not None:
        for opt in args.each:
            for value in opt:
                key = value.split('=')[0]
                val = value.split('=')[1].strip()
                each_engine_option.update({key: val})

    # Update tc of e1/e2 from each.
    if e1['tc'] == '' or e2['tc'] == '':
        if 'tc' in each_engine_option:
            for key, val in each_engine_option.items():
                if key == 'tc':
                    e1.update({key: val})
                    e2.update({key: val})
                    break

    # Update depth of e1/e2 from each.
    if e1['depth'] == 0 or e2['depth'] == 0:
        if 'depth' in each_engine_option:
            for key, val in each_engine_option.items():
                if key == 'depth':
                    e1.update({key: int(val)})
                    e2.update({key: int(val)})
                    break

    # Exit if there are no tc or depth.
    if e1['tc'] == '' or e2['tc'] == '':
        if e1['depth'] == 0 or e2['depth'] == 0:
            raise Exception('Error! tc or depth are not defined.')

    # Start opening file
    fen_file = None
    if args.openings is not None:
        for opt in args.openings:
            for value in opt:
                if 'file=' in value:
                    fen_file = value.split('=')[1]

    draw_option = {'movenumber': None, 'movecount': None, 'score': None}
    if args.draw is not None:
        for opt in args.draw[0]:
            key = opt.split('=')[0]
            val = int(opt.split('=')[1])
            draw_option.update({key: val})

    resign_option = {'movecount': None, 'score': None}
    if args.resign is not None:
        for opt in args.resign[0]:
            key = opt.split('=')[0]
            val = int(opt.split('=')[1])
            resign_option.update({key: val})

    is_random_startpos = True
    posround = 1  # Number of times the same position is played

    fens = get_fen_list(fen_file, is_random_startpos)

    output_game_file = args.pgnout

    # Start match
    joblist = []
    test_engine_score_list = []
    total_games = max(1, args.rounds // args.repeat)

    # Use Python 3.8 or higher
    with ProcessPoolExecutor(max_workers=args.concurrency) as executor:
        for i, fen in enumerate(fens if len(fens) else range(1000)):
            if i >= total_games:
                break
            job = executor.submit(round_match, fen, e1, e2,
                                  output_game_file, args.repeat,
                                  draw_option, resign_option, args.variant,
                                  posround)
            joblist.append(job)

        for future in concurrent.futures.as_completed(joblist):
            try:
                test_engine_score = future.result()[0]
                for s in test_engine_score:
                    test_engine_score_list.append(s)
                perf = mean(test_engine_score_list)
                games = len(test_engine_score_list)
                print(f'Score of {e1["name"]} vs {e2["name"]}: [{perf}] {games}')
            except concurrent.futures.process.BrokenProcessPool as ex:
                print(f'exception: {ex}')

    logging.info(f'final test score: {mean(test_engine_score_list)}')
    print('Finished match')


if __name__ == '__main__':
    main()
