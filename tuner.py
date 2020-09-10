import sys
from subprocess import Popen, PIPE
import copy
from collections import OrderedDict
import argparse
from pathlib import Path

import optuna


class Objective(object):
    def __init__(self, engine, param, best_param, init_value, variant,
                 opening_file, old_trial_num, base_time_sec=5,
                 inc_time_sec=0.05, rounds=16, concurrency=1, pgnout=None,
                 proto='uci', hashmb=64):
        self.param = copy.deepcopy(param)
        self.best_param = copy.deepcopy(best_param)
        self.best_result = init_value
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

        if len(self.best_param) == 0:
            for k, v in param.items():
                self.best_param.update({k: v[0]})

        self.trial_num = old_trial_num
        self.proto = proto
        self.hashmb = hashmb

    @staticmethod
    def set_param(from_param):
        new_param = {}
        for k, v in from_param.items():
            new_param.update({k: v[0]})

        return new_param

    def __call__(self, trial):
        print(f'starting trial: {self.trial_num} ...')

        # Options for test engine.
        test_options = ''
        for k, _ in self.param.items():
            par_val = trial.suggest_int(k, self.param[k][1], self.param[k][2])
            test_options += f'option.{k}={par_val} '
            self.test_param.update({k: par_val})
        test_options.rstrip()

        # Options for base engine.
        base_options = ''
        for k, v in self.best_param.items():
            base_options += f'option.{k}={v} '
        base_options.rstrip()

        print(f'suggested param: {self.test_param}')
        if self.trial_num > 0:
            print(f'best param: {self.best_param}')
            print(f'best value: {self.best_result}')
        else:
            print(f'init param: {self.best_param}')
            print(f'init value: {self.best_result}')

        tour_manager = Path(Path.cwd(), './tourney_manager/cutechess/cutechess-cli.exe')
        command = f' -concurrency {self.concurrency}'
        command += f' -engine cmd={self.e1} name={self.test_name} {test_options} proto={self.proto} option.Hash={self.hashmb}'
        command += f' -engine cmd={self.e2} name={self.base_name} {base_options} proto={self.proto} option.Hash={self.hashmb}'
        if self.variant != 'normal':
            command += f' -variant {self.variant}'
        command += f' -each tc=0/0:{self.base_time_sec}+{self.inc_time_sec}'
        command += f' -tournament round-robin'
        command += f' -rounds {self.rounds} -games 2 -repeat 2'
        command += f' -openings file={self.opening_file} format=epd'

        if self.pgnout is not None:
            command += f' -pgnout {self.pgnout}'

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

        # Update best param and value.
        if result > 0.5:
            inc = (result - 0.5) / 1000
            self.best_result += inc
            result = self.best_result

            for k, v in self.test_param.items():
                self.best_param.update({k: v})
        else:
            # If this is the first trial of a new study and it did not get
            # above 0.5, set the result to init value.
            if self.trial_num == 0:
                result = self.init_value

        self.trial_num += 1

        return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--engine', required=True,
                        help='Engine filename or engine path and filename.')
    parser.add_argument('--hash', required=False, type=int,
                        help='Engine memory in MB, default=64.', default=64)
    parser.add_argument('--trials', required=False, type=int,
                        help='Trials to try, default=1000.\n',
                        default=1000)
    parser.add_argument('--concurrency', required=False, type=int,
                        help='Number of game matches to run concurrently, default=1.\n',
                        default=1)
    parser.add_argument('--games-per-trial', required=False, type=int,
                        help='Number of games per trial, default=32.\n'
                        'This should be even number.', default=32)
    parser.add_argument('--study-name', required=False, type=str, default='default_study_name',
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

    args = parser.parse_args()

    trials = args.trials

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
    # Format, param name: [default, min, max]
    param = OrderedDict()
    param.update({'PawnValueOp': [0, 0, 1000], 'PawnValueEn': [0, 0, 1000]})
    param.update({'KnightValueOp': [0, 0, 1000], 'KnightValueEn': [0, 0, 1000]})
    param.update({'BishopValueOp': [0, 0, 1000], 'BishopValueEn': [0, 0, 1000]})
    param.update({'RookValueOp': [0, 0, 1000], 'RookValueEn': [0, 0, 1000]})
    param.update({'QueenValueOp': [0, 0, 2000], 'QueenValueEn': [0, 0, 2000]})

    # Define study.
    study = optuna.create_study(study_name=study_name, direction='maximize',
                                storage=f'sqlite:///{storage_file}',
                                load_if_exists=True)

    # Resume if there is existing data.
    try:
        init_best_param = copy.deepcopy(study.best_params)
    except ValueError:
        print('Warning, best param from previous trial is not found!, Use the init param.')
        init_best_param = Objective.set_param(param)
        print(f'init param: {init_best_param}')
    except:
        print('Unexpected error:', sys.exc_info()[0])
        raise
    else:
        print(f'best param: {init_best_param}')

    init_value = 0.5
    try:
        init_value = study.best_value
    except ValueError:
        print('Warning, init value is not found!')
        print(f'init best value: {init_value}')
    except:
        print('Unexpected error:', sys.exc_info()[0])
        raise
    else:
        print(f'best value: {init_value}')

    old_trial_num = len(study.trials)

    # Begin param optimization.
    study.optimize(Objective(args.engine, param,
                             init_best_param, init_value, variant,
                             opening_file, old_trial_num, base_time_sec,
                             inc_time_sec, rounds, args.concurrency,
                             pgnout, proto, args.hash),
                   n_trials=trials)

    # Show the best param, value and trial number.
    print()
    print(f'best param: {study.best_params}')
    print(f'best value: {study.best_value}')
    print(f'best trial number: {study.best_trial.number}')


if __name__ == "__main__":
    main()
