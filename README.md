# Optuna Game Parameter Tuner
A game search and evaluation parameter tuner using optuna framework. The game can be a chess or other game variants. Engine evaluation parameters that can be optimized are piece values like pawn value or knight value and others. Search parameters that can be optimized are futility pruning margin, null move reduction factors and others. 

## Optimization plots
![optimization_hist](https://camo.githubusercontent.com/4b10ec65d7b90f9ddac8b34e742b8278082ee5bf/68747470733a2f2f692e696d6775722e636f6d2f446877454652332e706e67)
***
![importances](https://camo.githubusercontent.com/e6111720a20e9d388098301e266ed5e357b99945/68747470733a2f2f692e696d6775722e636f6d2f326c684c7739592e706e67)
***
![slice](https://camo.githubusercontent.com/64444f11e3e03486b116af23da69f1dade6be96c/68747470733a2f2f692e696d6775722e636f6d2f774d32433341612e706e67)
***
![coordinate](https://camo.githubusercontent.com/fb2fef71e34d9db89140613202e0b57954d4cc63/68747470733a2f2f692e696d6775722e636f6d2f384473695835312e706e67)
***
![contour](https://camo.githubusercontent.com/debbbccaab8b714aea3789bddf3c15750098a13c/68747470733a2f2f692e696d6775722e636f6d2f4b533861704f652e706e67)

## Setup

#### Required
* Install python 3.8 or higher
  * Visit https://www.python.org/downloads/
* Install optuna
  * pip install optuna
  
#### Visualization
* pip install plotly
* pip install sklearn
* pip install kaleido

#### Save studies to pandas dataframe and csv file
* pip install pandas

#### To use skopt sampler
* pip install scikit-optimize

#### Install all dependencies
Instead of installing each module like optuna, plotly and others. Just install with requirements.txt.  
* pip install -r requirements.txt
  
## Basic optimization process outline
1. Prepare the engines and the parameters to be optimized. Set max_trial to 1000 or so.
2. Setup a game match between 2 engines. Test engine and base engine. Test engine will use the parameters suggested by optuna optimizer while base engine will use the initial default parameter values. In the beginning the best parameter is the initial parameter. The test engine will use the initial parameter values suggested by the optimimzer.
3. After a match of say 24 games, the score of test engine will be sent to the optimizer. The score is just (wins+draw/2)/24 from the point of view of the test engine. If the test engine wins (score > 0.5), update the best parameter to the parameter used by test engine. Increment trial by 1.
4. Check the number of trials done. If trial >= max_trial stop the optimization. This is done by the optimizer.
5. Get the new parameter values suggested by the optimizer. This will be used by the test engine.
6. Run a new match  
  a. The base engine will use the best parameter while the test engine will use the new parameter values suggested by the optimizer.  
  b. The base engine will always use the initial or default parameter values while the test engine will use the new parameter values suggested by the optimizer.
7. Goto step 3.
8. When max_trial is reached, optimization is stopped and png plots will be saved.
9. You can extend the trials or optimization by running the study again using the same study_name.

## Optimization strategy
It is an [optimization](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize) with default [TPE](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler) (Tree-structured Parzen Estimator Approach) as surrogate model or sampler. Optuna has some [samplers](https://optuna.readthedocs.io/en/stable/reference/samplers.html) that can be used in the optimization. Currently only the default TPE and an optional [CMAES](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler) and [skopt](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.SkoptSampler.html) are supported by this tuner.

## Help
```c++
python tuner.py -h
usage: Optuna Game Parameter Tuner v0.17.0 [-h] --engine ENGINE [--hash HASH] [--trials TRIALS] [--concurrency CONCURRENCY]
                                           [--games-per-trial GAMES_PER_TRIAL] [--study-name STUDY_NAME] [--base-time-sec BASE_TIME_SEC]
                                           [--inc-time-sec INC_TIME_SEC] [--depth DEPTH] --opening-file OPENING_FILE [--variant VARIANT]
                                           [--pgn-output PGN_OUTPUT] [--plot] [--initial-best-value INITIAL_BEST_VALUE]
                                           [--save-plots-every-trial SAVE_PLOTS_EVERY_TRIAL] [--fix-base-param] [--match-manager MATCH_MANAGER]
                                           [--protocol PROTOCOL] [--sampler [name= [option_name= ...]]] [--direction {maximize,minimize}]
                                           [--threshold-pruner [result= [games= ...]]] --input-param INPUT_PARAM

Optimize parameter values of a game agent using optuna framework.

optional arguments:
  -h, --help            show this help message and exit
  --engine ENGINE       Engine filename or engine path and filename.
  --hash HASH           Engine memory in MB, default=64.
  --trials TRIALS       Trials to try, default=1000.
  --concurrency CONCURRENCY
                        Number of game matches to run concurrently, default=1.
  --games-per-trial GAMES_PER_TRIAL
                        Number of games per trial, default=32.
                        This should be even number.
  --study-name STUDY_NAME
                        The name of study. This can be used to resume
                        study sessions, default=default_study_name.
  --base-time-sec BASE_TIME_SEC
                        Base time in sec for time control, default=5.
  --inc-time-sec INC_TIME_SEC
                        Increment time in sec for time control, default=0.05.
  --depth DEPTH         The maximum search depth that the engine is allowed, default=1000.
                        Example:
                        python tuner.py --depth 6 ...
                        If depth is high say 24 and you want this depth
                        to be always respected increase the base time control.
                        tuner.py --depth 24 --base-time-sec 300 ...
  --opening-file OPENING_FILE
                        Start opening filename in pgn, fen or epd format.
                        If match manager is cutechess, you can use pgn, fen
                        or epd format. The format is hard-coded currently.
                        You have to modify the code.
  --variant VARIANT     Game variant, default=normal.
  --pgn-output PGN_OUTPUT
                        Output pgn filename, default=optuna_games.pgn.
  --plot                A flag to output plots in png.
  --initial-best-value INITIAL_BEST_VALUE
                        The initial best value for the initial best
                        parameter values, default=0.5.
  --save-plots-every-trial SAVE_PLOTS_EVERY_TRIAL
                        Save plots every n trials, default=10.
  --fix-base-param      A flag to fix the parameter of base engine.
                        It will use the init or default parameter values.
  --match-manager MATCH_MANAGER
                        The application that handles the engine match, default=cutechess.
  --protocol PROTOCOL   The protocol that the engine supports, can be uci or cecp, default=uci.
  --sampler [name= [option_name= ...]]
                        The sampler to be used in the study, default name=tpe.
                        name can be tpe or cmaes or skopt, examples:
                        --sampler name=tpe ei_samples=50 ...
                          default ei_samples=24
                        --sampler name=cmaes ...
                        --sampler name=skopt acquisition_function=LCB ...
                          default acquisition_function=gp_hedge
                          Can be LCB or EI or PI or gp_hedge
  --direction {maximize,minimize}
                        The choice of whether to maximize or minimize the objective value to get the desired parameter values. default=maximize
  --threshold-pruner [result= [games= ...]]
                        A trial pruner used to prune or stop unpromising trials.
                        Example:
                        tuner.py --threshold-pruner result=0.45 games=50 interval=1 ...
                        Assuming games per trial is 100, after 50 games, check
                        the score of the match, if this is below 0.45, then
                        prune the trial or stop the engine match. Get new param
                        from optimizer and start a new trial.
                        Default values:
                        result=0.45, games=games_per_trial/2, interval=1
                        Example:
                        tuner.py --threshold-pruner ...
  --input-param INPUT_PARAM
                        The parameters that will be optimized.
                        Example 1 with 1 parameter:
                        --input-param "{'pawn': {'default': 92, 'min': 90, 'max': 120, 'step': 2}}"
                        Example 2 with 2 parameters:
                        --input-param "{'pawn': {'default': 92, 'min': 90, 'max': 120, 'step': 2}, 'knight': {'default': 300, 'min': 250, 'max': 350, 'step': 2}}"

Optuna Game Parameter Tuner v0.17.0
```

## Sample command line
```python
python tuner.py --engine ./engines/deuterium/deuterium --hash 128 --concurrency 6 --opening-file ./start_opening/ogpt_chess_startpos.epd --input-param "{'PawnValueEn': {'default':92, 'min':90, 'max':120, 'step':2}, 'BishopValueOp': {'default':350, 'min':290, 'max':350, 'step':3}, 'BishopValueEn': {'default':350, 'min':290, 'max':350, 'step':3}, 'RookValueEn': {'default':525, 'min':480, 'max':550, 'step':5}, 'QueenValueOp': {'default':985, 'min':950, 'max':1200, 'step':5}}" --initial-best-value 0.54 --games-per-trial 200 --plot --base-time-sec 120 --inc-time-sec 0.1 --depth 4 --study-name pv_d4_eisample_50_pruner --pgn-output train_pv_d4_eisamples_50_pruner.pgn --trials 200 --threshold-pruner result=0.35 --sampler name=skopt acquisition_function=LCB
```


## Optimization studies

* [Chess Piece Value Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization)
* [Chess Evaluation Positional Parameter Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-Evaluation-Positional-Parameter-Optimization)
* [Search Parameter Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Search-Parameter-Optimization)
* [Optimization with Threshold Pruner](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/commit/eb595ecb7a752cf2db6d8752b7480c59f696c7b7#commitcomment-42769655)
* [Optimization Performance Comparison](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/commit/d375e21af2f7b0bccdc11f215edf960628462c86#comments)

## Credits
* Optuna  
https://github.com/optuna/optuna
* Cutechess  
https://github.com/cutechess/cutechess
* Plotly  
https://plotly.com/
* Sklearn  
https://scikit-learn.org/stable/
