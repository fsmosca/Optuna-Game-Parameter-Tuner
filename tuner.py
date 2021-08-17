#!/usr/bin/env python


"""Optuna Game Parameter Tuner

Game parameter tuner using optuna framework. The game can be a chess or
chess variants. Parameters can be piece values for evaluations or
futility pruning margin for search."""


__author__ = 'fsmosca'
__script_name__ = 'Optuna Game Parameter Tuner'
__version__ = 'v0.41.1'
__credits__ = ['joergoster', 'musketeerchess', 'optuna']


import sys
from subprocess import Popen, PIPE
import copy
from collections import OrderedDict
import argparse
from pathlib import Path
import ast
from typing import List, Union
import logging

import optuna
from optuna.distributions import IntUniformDistribution, DiscreteUniformDistribution


logging.basicConfig(
    filename='log_tuner.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5.5s | %(message)s'
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.DEBUG)

chandler = logging.StreamHandler(sys.stdout)
chandler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-5.5s | %(message)s'))
chandler.setLevel(logging.INFO)
logger.addHandler(chandler)


is_panda_ok = True
try:
    import pandas
except ModuleNotFoundError:
    is_panda_ok = False
    logger.info('Warning! pandas is not installed.')


class Objective(object):
    def __init__(self, engine, input_param, best_param, best_value,
                 init_param, init_value, variant, opening_file,
                 opening_format, old_trial_num, pgnout,
                 nodes: Union[None, int]=None, base_time_sec=5,
                 inc_time_sec=0.05, rounds=16,
                 concurrency=1, proto='uci', fix_base_param=False,
                 match_manager='cutechess', good_result_cnt=0,
                 depth: Union[None, int]=None, games_per_trial=32,
                 threshold_pruner={}, common_param=None,
                 resign_movecount=None, resign_score=None,
                 draw_movenumber=None, draw_movecount=6, draw_score=0):
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
        self.opening_format = opening_format
        self.pgnout = pgnout
        self.nodes = nodes

        self.base_time_sec = base_time_sec
        self.inc_time_sec = inc_time_sec

        self.old_trial_num = old_trial_num

        self.test_param = {}

        self.trial_num = old_trial_num
        self.proto = proto

        # Todo: Improve inc_factor, 64 can relate to number of trials.
        self.inc_factor = 1/64
        self.fix_base_param = fix_base_param
        self.good_result_cnt = good_result_cnt
        self.match_manager = match_manager
        self.depth = depth
        self.games_per_trial = games_per_trial

        self.startup_trials = 10
        self.threshold_pruner = copy.deepcopy(threshold_pruner)

        if self.match_manager == 'cutechess' and self.proto == 'cecp':
            self.proto = 'xboard'

        self.common_param = common_param
        self.resign_movecount = resign_movecount
        self.resign_score = resign_score
        self.draw_movenumber = draw_movenumber
        self.draw_movecount = draw_movecount
        self.draw_score = draw_score

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
        else:
            logger.exception(f'Error, match_manager {match_man} is not supported.')
            raise

        return result

    @staticmethod
    def set_param(input_param):
        """
        Create a dict of default values from input param
        with default, min, max, and step.
        """
        new_param = {}
        for k, v in input_param.items():
            new_param.update({k: v['default']})

        return new_param

    def get_match_commands(self, test_options, base_options, games):
        if self.match_manager == 'cutechess':
            tour_manager = Path(Path.cwd(), './tourney_manager/cutechess/cutechess-cli.exe')
        else:
            tour_manager = 'python -u ./tourney_manager/duel/duel.py'

        command = f' -concurrency {self.concurrency}'

        if self.match_manager == 'cutechess':
            command += f' -engine cmd={self.e1} name={self.test_name} {test_options} proto={self.proto}'
            command += f' -engine cmd={self.e2} name={self.base_name} {base_options} proto={self.proto}'
            command += ' -wait 100'
        else:
            command += f' -engine cmd={self.e1} name={self.test_name} {test_options}'
            command += f' -engine cmd={self.e2} name={self.base_name} {base_options}'

        if self.variant != 'normal':
            command += f' -variant {self.variant}'

        command += ' -tournament round-robin'

        if self.match_manager == 'cutechess':
            command += ' -recover'
            command += f' -rounds {games//2} -games 2 -repeat 2'

            if self.depth is not None:
                command += f' -each tc=inf depth={self.depth}'
            elif self.nodes is not None:
                command += f' -each tc=inf nodes={self.nodes}'
            else:
                command += f' -each tc=0/0:{self.base_time_sec}+{self.inc_time_sec}'
        # duel.py match manager
        else:
            command += f' -rounds {games//2} -repeat 2'
            if self.depth is not None:
                command += f' -each tc=inf depth={self.depth}'
            else:
                command += f' -each tc=0/0:{self.base_time_sec}+{self.inc_time_sec}'

        if self.match_manager == 'cutechess':
            command += f' -openings file={self.opening_file} order=random format={self.opening_format}'
        else:
            command += f' -openings file={self.opening_file}'

        # draw adjudication
        if self.draw_movenumber is not None:
            command += f' -draw movenumber={self.draw_movenumber} movecount={self.draw_movecount} score={self.draw_score}'

        if self.resign_movecount is not None and self.resign_score is not None:
            command += f' -resign movecount={self.resign_movecount} score={self.resign_score}'
            if self.match_manager == 'cutechess':
                command += f' twosided=true'

        command += f' -pgnout {self.pgnout}'

        return tour_manager, command

    def engine_match(self, test_options, base_options, games=50) -> float:
        result = ''

        tour_manager, command = self.get_match_commands(
            test_options, base_options, games)

        # Execute the command line to start the match.
        process = Popen(str(tour_manager) + command, stdout=PIPE, text=True)
        for eline in iter(process.stdout.readline, ''):
            line = eline.strip()
            if line.startswith(f'Score of {self.test_name} vs {self.base_name}'):
                result = self.read_result(line)
                if 'Finished match' in line:
                    break

        if result == '':
            raise Exception('Error, there is something wrong with the engine match.')

        return result

    @staticmethod
    def result_mean(data: List[float]) -> float:
        return sum(data)/len(data)

    @staticmethod
    def get_sampler(args_sampler):
        if args_sampler is None:
            logger.warning('Sampler option is not defined, use tpe sampler.')
            return optuna.samplers.TPESampler()

        name = None
        for opt in args_sampler:
            for value in opt:
                if 'name=' in value:
                    name = value.split('=')[1]
                    break

        if name is None:
            logger.warning('Sampler name is not defined, use tpe sampler.')
            return optuna.samplers.TPESampler()

        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
        if name == 'tpe':
            ei_samples, multivariate, group, seed, constant_liar = 24, False, False, None, False
            for opt in args_sampler:
                for value in opt:
                    if 'ei_samples=' in value:
                        ei_samples = int(value.split('=')[1])
                    elif 'multivariate=' in value:
                        multivariate = True if value.split('=')[1].lower() == 'true' else False
                    elif 'group=' in value:
                        group = True if value.split('=')[1].lower() == 'true' else False
                    elif 'seed=' in value:
                        seed = int(value.split('=')[1])
                    elif 'constant_liar=' in value:
                        constant_liar = True if value.split('=')[1].lower() == 'true' else False

            # Avoid ValueError when multivariate is false and group is true
            if group and not multivariate:
                group = False
                logger.warning(f'group is set to false as multivariate is false!')

            return optuna.samplers.TPESampler(
                n_ei_candidates=ei_samples,  multivariate=multivariate,
                group=group, seed=seed, constant_liar=constant_liar)

        if name == 'cmaes':
            sigma0 = None  # initial std. deviation
            for opt in args_sampler:
                for value in opt:
                    if 'sigma0=' in value:
                        sigma0 = float(value.split('=')[1])

            # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.PyCmaSampler.html
            return optuna.integration.PyCmaSampler(sigma0=sigma0)

        if name == 'skopt':
            # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.SkoptSampler.html

            # Check acquisition function, It can be:
            # LCB, or EI, or PI or the default gp_hedge
            # https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer
            skopt_kwargs = {'acq_func': 'gp_hedge'}
            consider_pruned_trials = True

            af_value = ''
            for opt in args_sampler:
                for value in opt:
                    if 'acquisition_function=' in value:
                        af_value: str = value.split('=')[1]

                        if af_value in ['LCB', 'EI', 'PI', 'gp_hedge']:
                            skopt_kwargs.update({'acq_func': value.split('=')[1]})
                        else:
                            logger.exception(f'Error! acquisition function {af_value} is not supported. Use LCB or EI or PI or gp_hedge.')
                            raise
                    elif 'consider_pruned_trials=' in value:
                        consider_pruned_trials = True if value.split('=')[1].lower() == 'true' else False

            # Tweak exploration/exploitation.
            # LCB ->kappa, PI or EI ->xi
            # If kappa or xi is high, it favors exploration otherwise exploitation.
            # high: 10000, low: 0.0001
            # Ref.: https://scikit-optimize.github.io/stable/auto_examples/exploration-vs-exploitation.html#sphx-glr-auto-examples-exploration-vs-exploitation-py
            acq_func_kwargs = {}

            if af_value == 'LCB':
                for opt in args_sampler:
                    for value in opt:
                        if 'kappa=' in value:
                            acq_func_kwargs.update({'acq_func_kwargs': {'kappa': float(value.split('=')[1])}})
                            break
            elif af_value == 'EI' or af_value == 'PI':
                for opt in args_sampler:
                    for value in opt:
                        if 'xi=' in value:
                            acq_func_kwargs.update({'acq_func_kwargs': {'xi': float(value.split('=')[1])}})
                            break

            if len(acq_func_kwargs) > 0:
                skopt_kwargs.update(acq_func_kwargs)

            # Add base_estimator options such as GP, RF, ET, GBRT, default=GP.
            # Ref.: https://scikit-optimize.github.io/stable/modules/generated/skopt.Optimizer.html#skopt.Optimizer
            for opt in args_sampler:
                for value in opt:
                    if 'base_estimator=' in value:
                        be: str = value.split('=')[1]

                        if be in ['GP', 'RF', 'ET', 'GBRT']:
                            skopt_kwargs.update({'base_estimator': be})
                        else:
                            logger.exception(f'Error! base_estimator {be} is not supported. Use GP or RF or ET or GBRT.'
                                             f' Or do not write base-estimator at all and it will use GP.')
                            raise
                        break

            logger.info(f'skopt_kwargs: {skopt_kwargs}, consider_pruned_trials: {consider_pruned_trials}')

            return optuna.integration.SkoptSampler(
                skopt_kwargs=skopt_kwargs,
                consider_pruned_trials=consider_pruned_trials)

        logger.exception(f'Error, sampler name "{name}" is not supported, use tpe or cmaes or skopt.')
        raise

    @staticmethod
    def get_pruner(args_threshold_pruner, games_per_trial):
        pruner, th_pruner = None, {}
        if args_threshold_pruner is not None:

            # Default if there is threshold pruner.
            th_pruner.update({'result': 0.25, 'games': games_per_trial // 2, 'interval': 1})

            for opt in args_threshold_pruner:
                for value in opt:
                    if 'result=' in value:
                        th_pruner.update({value.split('=')[0]: float(value.split('=')[1])})
                    elif 'games=' in value:
                        th_pruner.update({value.split('=')[0]: int(value.split('=')[1])})
                    elif 'interval=' in value:
                        th_pruner.update({value.split('=')[0]: int(value.split('=')[1])})

            logger.info(f'pruner name: threshold_pruner,'
                        f' result: {th_pruner["result"]},'
                        f' games: {th_pruner["games"]},'
                        f' interval: {th_pruner["interval"]}\n')

            pruner = optuna.pruners.ThresholdPruner(
                lower=th_pruner["result"], n_warmup_steps=th_pruner["games"],
                interval_steps=th_pruner["interval"])

        return pruner, th_pruner

    def __call__(self, trial):
        logger.info('')
        logger.info(f'starting trial: {trial.number} ...')

        # Options for test engine.
        test_options = ''
        for k, v in self.input_param.items():
            # If input is of float type.
            try:
                param_type = v['type']
            except KeyError:
                param_type = 'int'
            if param_type == 'float':
                par_val = trial.suggest_float(k, v['min'], v['max'], step=v['step'])
            # Otherwise use integer.
            else:
                par_val = trial.suggest_int(k, v['min'], v['max'], v['step'])
            test_options += f'option.{k}={par_val} '
            self.test_param.update({k: par_val})

        # Add common param. It should not be included in the test param.
        if self.common_param is not None:
            for k, v in self.common_param.items():
                test_options += f'option.{k}={v} '

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

        # Add common param. It should not be included in the test param.
        if self.common_param is not None:
            for k, v in self.common_param.items():
                base_options += f'option.{k}={v} '

        base_options.rstrip()

        # Log info to console.
        logger.info(f'suggested param for test engine: {self.test_param}')
        if self.fix_base_param:
            logger.info(f'param for base engine          : {self.init_param}')
        else:
            if self.best_value > self.init_value:
                logger.info(f'param for base engine          : {self.best_param}')
            else:
                logger.info(f'param for base engine          : {self.init_param}')

        if self.common_param is not None:
            logger.info(f'common param: {self.common_param}')

        logger.info(f'init param: {self.init_param}')
        logger.info(f'init objective value: {self.init_value}')
        logger.info(f'study best param: {self.best_param}')
        logger.info(f'study best objective value: {self.best_value}')

        # Run engine vs engine match.
        if (len(self.threshold_pruner)
                and self.trial_num > self.startup_trials):
            games_to_play = self.threshold_pruner['games']
            result, played_games, final_result = 0.0, 0, []

            while True:
                logger.info(f'games_to_play: {games_to_play}')
                cur_result = self.engine_match(test_options, base_options, games_to_play)

                played_games += games_to_play
                final_result.append(cur_result)
                result = Objective.result_mean(final_result)
                logger.info(f'played_games: {played_games}, result: {{intermediate: {cur_result}, average: {result}}}')

                trial.report(result, played_games)

                if trial.should_prune():
                    logger.info(f'status: pruned, trial: {self.trial_num},'
                                f' played_games: {played_games},'
                                f' total_games: {self.games_per_trial},'
                                f' current_result: {result}')
                    self.trial_num += 1
                    raise optuna.TrialPruned()

                if played_games >= self.games_per_trial:
                    break

                games_to_play = min(
                    self.games_per_trial - played_games,
                    self.threshold_pruner['games'] * self.threshold_pruner['interval']
                )

            result = Objective.result_mean(final_result)
        else:
            result = self.engine_match(test_options, base_options, self.games_per_trial)

        logger.info(f'Actual match result: {result}, point of view: optimizer suggested values')

        # Output for match manager.
        test_param = ''
        for k, v in self.test_param.items():
            test_param += f'option.{k}={v} '
        logger.info(f'test param: {test_param}')

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

        logger.info(f'result sent to optimizer: {result}')

        return result


def save_plots(study, study_name, input_param, is_plot=False):
    if not is_plot:
        return

    logger.info('Saving plots ...')

    trials = len(study.trials)

    bg = '#F7D0CA'

    # Make sure there is a visuals folder in the current working folder.
    pre_name = f'./visuals/{study_name}_{trials}'

    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(paper_bgcolor=bg)
    fig.write_image(f'{pre_name}_hist.png')

    fig = optuna.visualization.plot_slice(study, params=list(input_param.keys()))
    fig.update_layout(paper_bgcolor=bg)
    fig.write_image(f'{pre_name}_slice.png')

    fig = optuna.visualization.plot_contour(study, params=list(input_param.keys()))
    if len(input_param) >= 3:
        fig.update_layout(width=1000, height=1000)
    fig.update_layout(paper_bgcolor=bg)
    fig.write_image(f'{pre_name}_contour.png')

    fig = optuna.visualization.plot_parallel_coordinate(study, params=list(input_param.keys()))
    fig.update_layout(paper_bgcolor=bg)
    fig.write_image(f'{pre_name}_parallel.png')

    fig = optuna.visualization.plot_param_importances(study)
    fig.update_layout(paper_bgcolor=bg)
    fig.write_image(f'{pre_name}_importance.png')

    logger.info('Done saving plots.\n')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog='%s %s' % (__script_name__, __version__),
        description='Optimize parameter values of a game agent using optuna framework.',
        epilog='%(prog)s')
    parser.add_argument('--engine', required=True,
                        help='Engine filename or engine path and filename.')
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
                        help='The maximum depth that the engine is allowed to search.')
    parser.add_argument('--nodes', required=False, type=int,
                        help='The maximum nodes that the engine is allowed to search.\n'
                             'This is only applicable to cutechess match manager.')
    parser.add_argument('--opening-file', required=True, type=str,
                        help='Start opening filename in pgn, fen or epd format.\n'
                             'If match manager is cutechess, you can use pgn, fen\n'
                             'or epd format. The format is hard-coded currently.\n'
                             'You have to modify the code.')
    parser.add_argument('--opening-format', required=False, type=str,
                        help='Can be pgn, or epd for cutechess match manager,'
                             'default is pgn, for duel.py no need as it will use epd or fen.',
                        default='pgn')
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
                        help='The application that handles the engine match,'
                             ' default=cutechess.',
                        default='cutechess')
    parser.add_argument('--protocol', required=False, type=str,
                        help='The protocol that the engine supports, can be'
                             ' uci or cecp, default=uci.',
                        default='uci')
    parser.add_argument('--sampler', required=False, nargs='*', action='append',
                        metavar=('name=', 'option_name='),
                        help='The sampler to be used in the study, default name=tpe.\n'
                             'name can be tpe or cmaes or skopt, examples:\n'
                             '--sampler name=tpe ei_samples=50 multivariate=true group=true seed=100 constant_liar=true  ...\n'
                             '  default values: ei_samples=24, multivariate=false, group=false, seed=None, constant_liar=false\n'
                             '  TPE ref: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html\n'
                             '--sampler name=cmaes sigma0=20 ...\n'
                             '  default sigma0 or initial std deviation is None. This tells cmaes that optimal parameter values\n'
                             '  lies within init_value +/- 3 * sigma0. By default this value is the parameter minimum_range/6.\n'
                             '  ref: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.PyCmaSampler.html\n'
                             '--sampler name=skopt acquisition_function=LCB ...\n'
                             '  default acquisition_function=gp_hedge\n'
                             '  Can be LCB or EI or PI or gp_hedge\n'
                             '  Example to explore, with LCB and kappa, high kappa would explore, low would exploit:\n'
                             '  --sampler name=skopt acquisition_function=LCB kappa=10000\n'
                             '  Example to exploit, with EI or PI and xi, high xi would explore, low would exploit:\n'
                             '  --sampler name=skopt acquisition_function=EI xi=0.0001\n'
                             '  Note: negative xi does not work with PI, but will work with EI.\n'
                             '  Ref.: https://scikit-optimize.github.io/stable/auto_examples/exploration-vs-exploitation.html#sphx-glr-auto-examples-exploration-vs-exploitation-py\n'
                             '  skopt has base_estimator options namely: GP, RF, ET and GBRT, default is GP.\n'
                             '  GP=Gaussian Process, RF=Random Forest, ET=Extra Tree, GBRT=Gradient Boosted Regressor Tree.\n'
                             '  Example:\n'
                             '  --sampler name=skopt base_estimator=GBRT acquisition_function=EI ...\n'
                             '  skopt has also a consider_pruned_trials parameter which is true by default. To not consider pruned trials use:\n'
                             '  --sampler name=skopt consider_pruned_trials=false ...\n'
                             '  consider_pruned_trials means that during sampling or finding the next best param values, the parameters\n'
                             '  that failed will be taken into account.')
    parser.add_argument('--threshold-pruner', required=False, nargs='*', action='append',
                        metavar=('result=', 'games='),
                        help='A trial pruner used to prune or stop unpromising'
                             ' trials.\n'
                             'Example:\n'
                             'tuner.py --threshold-pruner result=0.45 games=50 interval=1 ...\n'
                             'Assuming games per trial is 100, after 50 games, check\n'
                             'the score of the match, if this is below 0.45, then\n'
                             'prune the trial or stop the engine match. Get new param\n'
                             'from optimizer and start a new trial.\n'
                             'Default values:\n'
                             'result=0.25, games=games_per_trial/2, interval=1\n'
                             'Example:\n'
                             'tuner.py --threshold-pruner ...',
                        default=None)
    parser.add_argument('--input-param', required=True, type=str,
                        help='The parameters that will be optimized.\n'
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
    parser.add_argument('-v', '--version', action='version', version=f'{__version__}')
    parser.add_argument('--common-param', required=False, type=str,
                        help='The parameters that will be sent to both test and base engines.\n'
                             'Make sure that this param is not included in the input-param.\n'
                             'Example:\n'
                             '--common-param \"{\'RookOpenFile\': 92, \'KnightOutpost\': 300}\"')

    args = parser.parse_args()

    # Check if engine file exists.
    eng_path = Path(args.engine)
    if not eng_path.is_file():
        raise Exception(f'The engine in {eng_path} is missing!')

    trials = args.trials
    init_value = args.initial_best_value
    save_plots_every_trial = args.save_plots_every_trial
    fix_base_param = args.fix_base_param
    common_param = args.common_param

    if common_param is not None:
        common_param = ast.literal_eval(common_param)

    # Number of games should be even for a fair engine match.
    games_per_trial = args.games_per_trial
    games_per_trial += 1 if (args.games_per_trial % 2) != 0 else 0
    rounds = games_per_trial//2

    good_result_cnt = 0

    study_name = args.study_name
    storage_file = f'{study_name}.db'

    logger.info(f'{__script_name__} {__version__}')
    logger.info(f'trials: {trials}, games_per_trial: {rounds * 2}, sampler: {args.sampler}\n')

    # Convert the input param string to a dict of dict and sort by key.
    input_param = ast.literal_eval(args.input_param)
    input_param = OrderedDict(sorted(input_param.items()))

    logger.info(f'input param: {input_param}\n')
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
    sampler = Objective.get_sampler(args.sampler)

    # ThresholdPruner as trial pruner, if result of a match is below result
    # threshold after games threshold then prune the trial. Get new param
    # from optimizer and continue with the next trial.
    # --threshold-pruner result=0.45 games=50 --games-per-trial 100 ...
    pruner, th_pruner = Objective.get_pruner(args.threshold_pruner, games_per_trial)

    logger.info('Starting optimization ...')

    while cycle < max_cycle:
        cycle += 1

        # Define study.
        study = optuna.create_study(study_name=study_name,
                                    direction='maximize',
                                    storage=f'sqlite:///{storage_file}',
                                    load_if_exists=True, sampler=sampler,
                                    pruner=pruner)

        # Get the best value from previous study session.
        best_param, best_value, is_study = {}, 0.0, False
        try:
            best_value = study.best_value
            is_study = True
        except ValueError:
            logger.warning('Warning, best value from previous trial is not found!')
        except:
            logger.exception('Unexpected error:', sys.exc_info()[0])
            raise
        logger.info(f'study best objective value: {best_value}')

        # Get the best param values from previous study session.
        try:
            best_param = copy.deepcopy(study.best_params)
        except ValueError:
            logger.warning('Warning, best param from previous trial is not found!.')
        except:
            logger.exception('Unexpected error:', sys.exc_info()[0])
            raise
        logger.info(f'study best param: {best_param}')

        old_trial_num = len(study.trials)

        # Get the good result count before we resume the study.
        if is_panda_ok and not fix_base_param and is_study:
            df = study.trials_dataframe(attrs=('value', 'state'))
            for index, row in df.iterrows():
                if row['value'] > init_value and row['state'] == 'COMPLETE':
                    good_result_cnt += 1

        # If there is no trial recorded yet we will initialize our study
        # with default values from the engine.
        if not is_study:
            distri = {}
            init_trial_value = init_value

            for k, v in input_param.items():
                if 'type' in v and v['type'] == 'float':
                    distri.update({k: DiscreteUniformDistribution(v['min'], v['max'], v['step'])})
                else:
                    distri.update({k: IntUniformDistribution(v['min'], v['max'], v['step'])})

            init_trial = optuna.trial.create_trial(
                params=copy.deepcopy(init_param),
                distributions=copy.deepcopy(distri),
                value=init_trial_value,
            )
            study.add_trial(init_trial)

            logger.info(f"added trial: yes, params: {init_trial.params}, objective value: {init_trial.values[0]}")

            best_param = study.best_params
            best_value = study.best_value

        # Begin param optimization.
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        study.optimize(Objective(args.engine, input_param, best_param,
                                 best_value, init_param, init_value,
                                 args.variant, args.opening_file,
                                 args.opening_format, old_trial_num,
                                 args.pgn_output, args.nodes,
                                 args.base_time_sec, args.inc_time_sec,
                                 rounds, args.concurrency,
                                 args.protocol, fix_base_param,
                                 args.match_manager, good_result_cnt,
                                 args.depth, games_per_trial, th_pruner,
                                 common_param, args.resign_movecount,
                                 args.resign_score, args.draw_movenumber,
                                 args.draw_movecount, args.draw_score),
                       n_trials=n_trials)

        # Create and save plots after this study session is completed.
        save_plots(study, study_name, input_param, args.plot)

        # Build pandas dataframe, print and save to csv file.
        if is_panda_ok:
            df = study.trials_dataframe(attrs=('number', 'value', 'params',
                                               'state'))
            logger.info(f'{df.to_string(index=False)}\n')
            df.to_csv(f'{study_name}.csv', index=False)

        # Show the best param, value and trial number.
        logger.info(f'study best param: {study.best_params}')
        logger.info(f'study best objective value: {study.best_value}')
        logger.info(f'study best trial number: {study.best_trial.number}\n')

        # Output for match manager.
        option_output = ''
        for k, v in study.best_params.items():
            option_output += f'option.{k}={v} '
        logger.info(f'{option_output}\n')


if __name__ == "__main__":
    main()
