import sys
from subprocess import Popen, PIPE
import copy
from collections import OrderedDict
import argparse
from pathlib import Path

import optuna

is_panda_ok = True
try:
    import pandas
except ModuleNotFoundError:
    is_panda_ok = False
    print('Warning! pandas is not installed.')


APP_NAME = 'Optuna Game Parameter Tuner'
APP_VERSION = 'v0.2.0'


class Objective(object):
    def __init__(self, engine, input_param, best_param, best_value,
                 init_param, init_value, variant, opening_file, old_trial_num,
                 base_time_sec=5, inc_time_sec=0.05, rounds=16,
                 concurrency=1, pgnout=None, proto='uci', hashmb=64):
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

        self.inc_factor = 1/64

    @staticmethod
    def set_param(from_param):
        new_param = {}
        for k, v in from_param.items():
            new_param.update({k: v['default']})

        return new_param

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
        if self.best_value > self.init_value:
            for k, v in self.best_param.items():
                base_options += f'option.{k}={v} '
        else:
            for k, v in self.init_param.items():
                base_options += f'option.{k}={v} '
        base_options.rstrip()

        # Log info to console.
        print(f'suggested param for test engine: {self.test_param}')
        if self.best_value > self.init_value:
            print(f'param for base engine          : {self.best_param}')
        else:
            print(f'param for base engine          : {self.init_param}')

        print(f'init param: {self.init_param}')
        print(f'init value: {self.init_value}')
        print(f'study best param: {self.best_param}')
        print(f'study best value: {self.best_value}')

        # Create command line for the engine match using cutechess-cli.
        tour_manager = Path(Path.cwd(), './tourney_manager/cutechess/cutechess-cli.exe')
        command = f' -concurrency {self.concurrency}'
        command += f' -engine cmd={self.e1} name={self.test_name} {test_options} proto={self.proto} option.Hash={self.hashmb}'
        command += f' -engine cmd={self.e2} name={self.base_name} {base_options} proto={self.proto} option.Hash={self.hashmb}'
        if self.variant != 'normal':
            command += f' -variant {self.variant}'
        command += f' -each tc=0/0:{self.base_time_sec}+{self.inc_time_sec}'
        command += ' -tournament round-robin'
        command += f' -rounds {self.rounds} -games 2 -repeat 2'
        command += f' -openings file={self.opening_file} format=epd'
        command += ' -resign movecount=6 score=700 twosided=true'
        command += ' -draw movenumber=30 movecount=6 score=5'
        command += f' -pgnout {self.pgnout}'

        # Execute the command line to start the match.
        process = Popen(str(tour_manager) + command, stdout=PIPE, text=True)
        output = process.communicate()[0]
        if process.returncode != 0:
            print('Could not execute command: %s' % command)
            return 2

        result = ""
        for line in output.splitlines():
            if line.startswith(f'Score of {self.test_name} vs {self.base_name}'):
                result = line[line.find("[") + 1: line.find("]")]

        if result == "":
            raise Exception('The match did not terminate properly!')

        result = float(result)
        print(f'Actual match result: {result}, point of view: optimizer suggested values')

        # Update best param and value. We modify the result here because the
        # optimizer will consider the max result in its algorithm.
        # Ref.: https://github.com/optuna/optuna/issues/1728
        if result > self.init_value:
            if self.best_value < self.init_value:
                self.best_value = self.init_value
            inc = self.inc_factor * (result - self.init_value)
            self.best_value += inc
            result = self.best_value

            for k, v in self.test_param.items():
                self.best_param.update({k: v})
        else:
            if result > self.best_value:
                self.best_value = result

                for k, v in self.test_param.items():
                    self.best_param.update({k: v})

        self.trial_num += 1

        return result


def save_plots(study, study_name, input_param, is_plot=False):
    if not is_plot:
        return

    print('Saving plots ...')

    trials = len(study.trials)

    # Make sure there is a visuals folder in the current working folder.
    # Todo: Make an option with default value visuals
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
        prog='%s %s' % (APP_NAME, APP_VERSION),
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
    parser.add_argument('--opening-file', required=True, type=str,
                        help='Start opening filename in fen or epd format.')
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

    args = parser.parse_args()

    trials = args.trials
    init_value = args.initial_best_value
    save_plots_every_trial = args.save_plots_every_trial

    # Number of games should be even for a fair engine match.
    num_games = args.games_per_trial
    num_games += 1 if (args.games_per_trial % 2) != 0 else 0
    rounds = num_games//2

    base_time_sec = args.base_time_sec
    inc_time_sec = args.inc_time_sec
    opening_file = args.opening_file
    variant = args.variant
    pgnout = args.pgn_output
    proto = 'uci'

    study_name = args.study_name
    storage_file = f'{study_name}.db'

    print(f'trials: {trials}, games_per_trial: {rounds * 2}')

    # Define the parameters that will be optimized.
    input_param = OrderedDict()
    input_param.update({'PawnValueEn': {'default': 92, 'min': 90, 'max': 120, 'step': 2}})
    input_param.update({'BishopValueOp': {'default': 350, 'min': 300, 'max': 360, 'step': 3}})
    input_param.update({'BishopValueEn': {'default': 350, 'min': 300, 'max': 360, 'step': 3}})
    input_param.update({'RookValueEn': {'default': 525, 'min': 490, 'max': 550, 'step': 5}})
    input_param.update({'QueenValueOp': {'default': 985, 'min': 975, 'max': 1050, 'step': 5}})
    input_param.update({'MobilityWeight': {'default': 100, 'min': 50, 'max': 150, 'step': 4}})

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

    while cycle < max_cycle:
        cycle += 1

        # Define study.
        study = optuna.create_study(study_name=study_name, direction='maximize',
                                    storage=f'sqlite:///{storage_file}',
                                    load_if_exists=True)

        # Get the best value from previous study session.
        best_param, best_value = {}, 0.0
        try:
            best_value = study.best_value
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

        # Begin param optimization.
        study.optimize(Objective(args.engine, input_param,
                                 best_param, best_value, init_param,
                                 init_value, variant, opening_file,
                                 old_trial_num, base_time_sec,
                                 inc_time_sec, rounds, args.concurrency,
                                 pgnout, proto, args.hash),
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
        print(f'best param: {study.best_params}')
        print(f'best value: {study.best_value}')
        print(f'best trial number: {study.best_trial.number}')


if __name__ == "__main__":
    main()
