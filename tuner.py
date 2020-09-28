#!/usr/bin/env python


"""Optuna Game Parameter Tuner

Game parameter tuner using optuna framework. The game can be a chess or
chess variants. Parameters can be piece values for evaluations or
futility pruning margin for search."""


__author__ = 'fsmosca'
__script_name__ = 'Optuna Game Parameter Tuner'
__version__ = 'v0.10.0'
__credits__ = ['joergoster', 'musketeerchess', 'optuna']


import sys
from subprocess import Popen, PIPE
import copy
from collections import OrderedDict
import argparse
from pathlib import Path
import ast

import optuna

is_panda_ok = True
try:
    import pandas
except ModuleNotFoundError:
    is_panda_ok = False
    print('Warning! pandas is not installed.')


class Objective(object):
    def __init__(self, engine, input_param, best_param, best_value,
                 init_param, init_value, variant, opening_file,
                 old_trial_num, pgnout, base_time_sec=5,
                 inc_time_sec=0.05, rounds=16, concurrency=1,
                 proto='uci', hashmb=64, fix_base_param=False,
                 match_manager='cutechess', good_result_cnt=0,
                 depth=1000, threshold_pruner_name='',
                 threshold_pruner_result=0.45,
                 threshold_pruner_games=16):
        self.input_param = copy.deepcopy(input_param)
        self.best_param = copy.deepcopy(best_param)
        self.best_value = best_value
        self.init_param = copy.deepcopy(init_param)
        self.init_value = init_value
        self.rounds = rounds

        self.variant = variant

        self.test_name = 'test'
        self.base_name = 'base'

        self.e1 = engine
        self.e2 = engine

        self.concurrency = concurrency
        self.opening_file = opening_file
        self.pgnout = pgnout

        self.base_time_sec = base_time_sec
        self.inc_time_sec = inc_time_sec

        self.old_trial_num = old_trial_num

        self.test_param = {}

        self.trial_num = old_trial_num
        self.proto = proto
        self.hashmb = hashmb

        # Todo: Improve inc_factor, 64 can relate to number of trials.
        self.inc_factor = 1/64
        self.fix_base_param = fix_base_param
        self.good_result_cnt = good_result_cnt
        self.match_manager = match_manager
        self.depth = depth
        self.games_per_trial = self.rounds * 2

        self.startup_trials = 10
        self.threshold_pruner_name = threshold_pruner_name
        self.threshold_pruner_result = threshold_pruner_result
        self.threshold_pruner_games = threshold_pruner_games

    def read_result(self, line: str) -> float:
        """Read result output line from match manager."""
        match_man = self.match_manager

        if match_man == 'cutechess':
            # Score of e1 vs e2: 39 - 28 - 64  [0.542] 131
            num_wins = int(line.split(': ')[1].split(' -')[0])
            num_draws = int(line.split(': ')[1].split('-')[2].strip().split()[0])
            num_games = int(line.split('] ')[1].strip())
            result = (num_wins + num_draws / 2) / num_games
        elif match_man == 'duel':
            # Score of e1 vs e2: [0.4575] 20
            result = float(line.split('[')[1].split(']')[0])
            num_games = int(line.split('] ')[1].strip())
        else:
            print(f'Error, match_manager {match_man} is not supported.')
            raise

        return result, num_games

    @staticmethod
    def set_param(from_param):
        new_param = {}
        for k, v in from_param.items():
            new_param.update({k: v['default']})

        return new_param

    def get_match_commands(self, test_options, base_options):
        if self.match_manager == 'cutechess':
            tour_manager = Path(Path.cwd(), './tourney_manager/cutechess/cutechess-cli.exe')
        else:
            tour_manager = f'python -u ./tourney_manager/duel/duel.py'

        command = f' -concurrency {self.concurrency}'

        if self.match_manager == 'cutechess':
            command += f' -engine cmd={self.e1} name={self.test_name} {test_options} proto={self.proto} option.Hash={self.hashmb}'
            command += f' -engine cmd={self.e2} name={self.base_name} {base_options} proto={self.proto} option.Hash={self.hashmb}'
        else:
            command += f' -engine cmd={self.e1} name={self.test_name} {test_options}'
            command += f' -engine cmd={self.e2} name={self.base_name} {base_options}'

        if self.variant != 'normal':
            command += f' -variant {self.variant}'

        command += ' -tournament round-robin'

        if self.match_manager == 'cutechess':
            command += f' -rounds {self.rounds} -games 2 -repeat 2'
            command += f' -each tc=0/0:{self.base_time_sec}+{self.inc_time_sec} depth={self.depth}'
        else:
            command += f' -rounds {self.rounds*2} -repeat 2'
            command += f' -each tc=0/0:{self.base_time_sec}+{self.inc_time_sec}'

        if self.match_manager == 'cutechess':
            command += f' -openings file={self.opening_file} format=epd'
            command += ' -resign movecount=6 score=700 twosided=true'
            command += ' -draw movenumber=30 movecount=6 score=5'
        else:
            command += f' -openings file={self.opening_file}'

        command += f' -pgnout {self.pgnout}'

        return tour_manager, command

    def __call__(self, trial):
        print()
        print(f'starting trial: {self.trial_num} ...')

        # Options for test engine.
        test_options = ''
        for k, v in self.input_param.items():
            par_val = trial.suggest_int(k, v['min'], v['max'], v['step'])
            test_options += f'option.{k}={par_val} '
            self.test_param.update({k: par_val})
        test_options.rstrip()

        # Options for base engine.
        base_options = ''
        if self.fix_base_param:
            for k, v in self.init_param.items():
                base_options += f'option.{k}={v} '
        else:
            if self.best_value > self.init_value:
                for k, v in self.best_param.items():
                    base_options += f'option.{k}={v} '
            else:
                for k, v in self.init_param.items():
                    base_options += f'option.{k}={v} '
        base_options.rstrip()

        # Log info to console.
        print(f'suggested param for test engine: {self.test_param}')
        if self.fix_base_param:
            print(f'param for base engine          : {self.init_param}')
        else:
            if self.best_value > self.init_value:
                print(f'param for base engine          : {self.best_param}')
            else:
                print(f'param for base engine          : {self.init_param}')

        print(f'init param: {self.init_param}')
        print(f'init value: {self.init_value}')
        print(f'study best param: {self.best_param}')
        print(f'study best value: {self.best_value}')

        # Run engine vs engine match.
        terminate_match, result = False, ''

        # Create command line for the engine match using cutechess-cli or duel.py.
        tour_manager, command = self.get_match_commands(test_options, base_options)

        # Execute the command line to start the match.
        process = Popen(str(tour_manager) + command, stdout=PIPE, text=True)
        for eline in iter(process.stdout.readline, ''):
            line = eline.strip()
            if line.startswith(f'Score of {self.test_name} vs {self.base_name}'):
                result, done_num_games = self.read_result(line)

                # Check if we will prune this trial.
                if (self.threshold_pruner_name != ''
                        and self.trial_num > self.startup_trials
                        and done_num_games > self.threshold_pruner_games
                        and result < self.threshold_pruner_result
                        and self.match_manager == 'cutechess'):
                    process.terminate()

                    while True:
                        process.wait(timeout=1)
                        if process.poll() is not None:
                            break

                    terminate_match = True
                    break
                else:
                    if 'Finished match' in line:
                        break

        if result == '':
            print('Error, there is something wrong with the engine match.')
            raise

        # Prune trials that are not promising. If number of games
        # per trial is 100 and after 50 games, its result is below 0.45 or 45%
        # then we stop this trial. Get new param values and start a new trial.
        trial.report(result, done_num_games)
        if terminate_match:
            if trial.should_prune():
                self.trial_num += 1
                print(f'status: prune, trial: {self.trial_num}, done_games: {done_num_games}, total_games:{self.rounds * 2}, current_result: {result}')
                raise optuna.TrialPruned()

        print(f'Actual match result: {result}, point of view: optimizer suggested values')

        # If base engine always uses the initial param or default param.
        if self.fix_base_param:
            # Backup best value and param.
            if result > self.best_value:
                self.best_value = result

                for k, v in self.test_param.items():
                    self.best_param.update({k: v})

        # Else if best param used by base engine is dynamic, meaning the base
        # engine will use the available best param as long as the best value
        # of this best param is better than the init value.
        else:
            # Update best param and value. We modify the result here because the
            # optimizer will consider the max result in its algorithm.
            # Ref.: https://github.com/optuna/optuna/issues/1728
            if result > self.init_value:
                self.good_result_cnt += 1
                if self.best_value < self.init_value:
                    self.best_value = self.init_value
                inc = self.inc_factor * (result - self.init_value)
                self.best_value += inc
                result = self.best_value

                for k, v in self.test_param.items():
                    self.best_param.update({k: v})
            else:
                # Backup study best value and param.
                if result > self.best_value:
                    self.best_value = result

                    for k, v in self.test_param.items():
                        self.best_param.update({k: v})

                # Adjust the result sent to the optimizer. Given a match
                # result of 0.48 from trial 0, good_result_cnt of 0 and then
                # later a match result of 0.48 at trial 50 with good_result_cnt
                # of 4, the latter performs better and their results are
                # different in the eyes of the optimizer.
                # Trial:  0, good_result_cnt: 0, actual_result: 0.48, result: 0.470
                # Trial: 50, good_result_cnt: 4, actual_result: 0.48, result: 0.478
                result = result - 0.01/(self.good_result_cnt + 1)

        self.trial_num += 1

        return result


def save_plots(study, study_name, input_param, is_plot=False):
    if not is_plot:
        return

    print('Saving plots ...')

    trials = len(study.trials)

    # Make sure there is a visuals folder in the current working folder.
    pre_name = f'./visuals/{study_name}_{trials}'

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f'{pre_name}_hist.png')

    fig = optuna.visualization.plot_slice(study, params=list(input_param.keys()))
    fig.write_image(f'{pre_name}_slice.png')

    fig = optuna.visualization.plot_contour(study, params=list(input_param.keys()))
    if len(input_param) >= 3:
        fig.update_layout(width=1000, height=1000)
    fig.write_image(f'{pre_name}_contour.png')

    fig = optuna.visualization.plot_parallel_coordinate(study, params=list(input_param.keys()))
    fig.write_image(f'{pre_name}_parallel.png')

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f'{pre_name}_importance.png')

    print('Done saving plots.')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog='%s %s' % (__script_name__, __version__),
        description='Optimize parameter values of a game agent using optuna framework.',
        epilog='%(prog)s')
    parser.add_argument('--engine', required=True,
                        help='Engine filename or engine path and filename.')
    parser.add_argument('--hash', required=False, type=int,
                        help='Engine memory in MB, default=64.', default=64)
    parser.add_argument('--trials', required=False, type=int,
                        help='Trials to try, default=1000.',
                        default=1000)
    parser.add_argument('--concurrency', required=False, type=int,
                        help='Number of game matches to run concurrently, default=1.',
                        default=1)
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
                        help='The maximum search depth that the engine is'
                             ' allowed, default=1000.\n'
                             'Example:\n'
                             'python tuner.py --depth 6 ...\n'
                             'If depth is high say 24 and you want this depth\n'
                             'to be always respected increase the base time'
                             ' control.\n'
                             'tuner.py --depth 24 --base-time-sec 300 ...',
                        default=1000)
    parser.add_argument('--opening-file', required=True, type=str,
                        help='Start opening filename in pgn, fen or epd format.\n'
                             'If match manager is cutechess, you can use pgn, fen\n'
                             'or epd format. The format is hard-coded currently.\n'
                             'You have to modify the code.')
    parser.add_argument('--variant', required=False, type=str,
                        help='Game variant, default=normal.', default='normal')
    parser.add_argument('--pgn-output', required=False, type=str,
                        help='Output pgn filename, default=optuna_games.pgn.',
                        default='optuna_games.pgn')
    parser.add_argument('--plot', action='store_true', help='A flag to output plots in png.')
    parser.add_argument('--initial-best-value', required=False, type=float,
                        help='The initial best value for the initial best\n'
                             'parameter values, default=0.5.', default=0.5)
    parser.add_argument('--save-plots-every-trial', required=False, type=int,
                        help='Save plots every n trials, default=10.',
                        default=10)
    parser.add_argument('--fix-base-param', action='store_true',
                        help='A flag to fix the parameter of base engine.\n'
                             'It will use the init or default parameter values.')
    parser.add_argument('--match-manager', required=False, type=str,
                        help='The application that handles the engine match, default=cutechess.',
                        default='cutechess')
    parser.add_argument('--protocol', required=False, type=str,
                        help='The protocol that the engine supports, can be uci or cecp, default=uci.',
                        default='uci')
    parser.add_argument('--sampler', required=False, type=str,
                        help='The sampler to be used in the study,'
                             ' default=tpe, can be tpe or cmaes.',
                        default='tpe')
    parser.add_argument('--trial-pruning', required=False, nargs='*', action='append',
                        metavar=('name=', 'result='),
                        help='A trial pruner used to prune or stop unpromising'
                             ' trials, default=None.\n'
                             'Example:\n'
                             'tuner.py --trial-pruning name=threshold_pruner result=0.45 games=50 ...\n'
                             'Assuming games per trial is 100, after 50 games, check\n'
                             'the score of the match, if this is below 0.45, then\n'
                             'prune the trial or stop the engine match. Get new param\n'
                             'from optimizer and start a new trial.\n'
                             'Default values:\n'
                             'result=0.45, games=games_per_trial/2\n'
                             'Example:\n'
                             'tuner.py --trial-pruning name=threshold_pruner ...',
                        default=None)
    parser.add_argument('--tpe-ei-samples', required=False, type=int,
                        help='The number of candidate samples used'
                             ' to calculate ei or expected improvement,'
                             ' default=24.',
                        default=24)
    parser.add_argument('--input-param', required=True, type=str,
                        help='The parameters that will be optimized.\n'
                             'Example 1 with 1 parameter:\n'
                             '--input-param \"{\'pawn\': {\'default\': 92, \'min\': 90, \'max\': 120, \'step\': 2}}\"\n'
                             'Example 2 with 2 parameters:\n'
                             '--input-param \"{\'pawn\': {\'default\': 92, \'min\': 90, \'max\': 120, \'step\': 2}, \'knight\': {\'default\': 300, \'min\': 250, \'max\': 350, \'step\': 2}}\"'
                        )

    args = parser.parse_args()

    trials = args.trials
    init_value = args.initial_best_value
    save_plots_every_trial = args.save_plots_every_trial
    fix_base_param = args.fix_base_param
    match_manager = args.match_manager
    args_sampler = args.sampler

    # Number of games should be even for a fair engine match.
    num_games = args.games_per_trial
    num_games += 1 if (args.games_per_trial % 2) != 0 else 0
    rounds = num_games//2

    base_time_sec = args.base_time_sec
    inc_time_sec = args.inc_time_sec
    opening_file = args.opening_file
    variant = args.variant
    pgnout = args.pgn_output
    proto = args.protocol

    good_result_cnt = 0

    study_name = args.study_name
    storage_file = f'{study_name}.db'

    print(f'trials: {trials}, games_per_trial: {rounds * 2}')

    # Convert the input param string to a dict of dict and sort by key.
    input_param = ast.literal_eval(args.input_param)
    input_param = OrderedDict(sorted(input_param.items()))

    print(f'input param: {input_param}\n')
    init_param = Objective.set_param(input_param)

    # Adjust save_plots_every_trial if trials is lower than it so
    # that max_cycle is 1 or more and studies can continue. The plot
    # will be generated after the study.
    if trials < save_plots_every_trial:
        save_plots_every_trial = trials

    max_cycle = trials // save_plots_every_trial
    n_trials = save_plots_every_trial
    cycle = 0

    # Define sampler to use, default is TPE.
    if args_sampler == 'tpe':
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
        sampler = optuna.samplers.TPESampler(n_ei_candidates=args.tpe_ei_samples)
    elif args_sampler == 'cmaes':
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html#
        sampler = optuna.samplers.CmaEsSampler()
    else:
        msg = f'The sampler {args_sampler} is not suppored. Use tpe or cmaes.'
        raise ValueError(msg)

    # Trial Pruner, if after half of games in a trial the result is below
    # 0.45, prune the trial. Take new param values from optimizer and start
    # a new trial.
    tp_name = ''
    tp_result = 0.45  # Prune if result is below this.
    tp_games = num_games // 2  # Prune if played games is above this.

    if args.trial_pruning is not None:
        for opt in args.trial_pruning:
            for value in opt:
                if 'name=' in value:
                    tp_name = value.split('=')[1]
                elif 'result=' in value:
                    tp_result = float(value.split('=')[1])
                elif 'games=' in value:
                    tp_games = int(value.split('=')[1])
        if tp_name == 'threshold_pruner':
            print(f'name: {tp_name}, result: {tp_result}, games: {tp_games}')
            pruner = optuna.pruners.ThresholdPruner(
                lower=tp_result, n_warmup_steps=tp_games, interval_steps=1)
        else:
            pruner = None
    else:
        pruner = args.trial_pruning

    while cycle < max_cycle:
        cycle += 1

        # Define study.
        study = optuna.create_study(study_name=study_name, direction='maximize',
                                    storage=f'sqlite:///{storage_file}',
                                    load_if_exists=True, sampler=sampler,
                                    pruner=pruner)

        # Get the best value from previous study session.
        best_param, best_value, is_study = {}, 0.0, False
        try:
            best_value = study.best_value
            is_study = True
        except ValueError:
            print('Warning, best value from previous trial is not found!')
        except:
            print('Unexpected error:', sys.exc_info()[0])
            raise
        print(f'study best value: {best_value}')

        # Get the best param values from previous study session.
        try:
            best_param = copy.deepcopy(study.best_params)
        except ValueError:
            print('Warning, best param from previous trial is not found!.')
        except:
            print('Unexpected error:', sys.exc_info()[0])
            raise
        print(f'study best param: {best_param}')

        old_trial_num = len(study.trials)

        # Get the good result count before we resume the study.
        if is_panda_ok and not fix_base_param and is_study:
            df = study.trials_dataframe(attrs=('value', 'state'))
            for index, row in df.iterrows():
                if row['value'] > init_value and row['state'] == 'COMPLETE':
                    good_result_cnt += 1

        # Begin param optimization.
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        study.optimize(Objective(args.engine, input_param,
                                 best_param, best_value, init_param,
                                 init_value, variant, opening_file,
                                 old_trial_num, pgnout, base_time_sec,
                                 inc_time_sec, rounds, args.concurrency,
                                 proto, args.hash, fix_base_param,
                                 match_manager, good_result_cnt, args.depth,
                                 tp_name, tp_result, tp_games),
                       n_trials=n_trials)

        # Create and save plots after this study session is completed.
        save_plots(study, study_name, input_param, args.plot)

        # Build pandas dataframe, print and save to csv file.
        if is_panda_ok:
            df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
            print(df.to_string(index=False))
            df.to_csv(f'{study_name}.csv', index=False)

        # Show the best param, value and trial number.
        print()
        print(f'study best param: {study.best_params}')
        print(f'study best value: {study.best_value}')
        print(f'study best trial number: {study.best_trial.number}')


if __name__ == "__main__":
    main()
