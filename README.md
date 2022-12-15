# Optuna Game Parameter Tuner
[![Python 3.8](https://img.shields.io/badge/Python-%203.8%20%7C%203.9%20-cyan.svg)](https://www.python.org/downloads/release/python-380/)

A game search and evaluation parameter tuner using optuna framework. The game can be a chess or other game variants. Engine evaluation parameters that can be optimized are piece values like pawn value or knight value and others. Search parameters that can be optimized are futility pruning margin, null move reduction factors and others. 

## A. Optimization plots
![optimization_hist](https://camo.githubusercontent.com/4b10ec65d7b90f9ddac8b34e742b8278082ee5bf/68747470733a2f2f692e696d6775722e636f6d2f446877454652332e706e67)
***
![importances](https://camo.githubusercontent.com/e6111720a20e9d388098301e266ed5e357b99945/68747470733a2f2f692e696d6775722e636f6d2f326c684c7739592e706e67)
***
![slice](https://camo.githubusercontent.com/64444f11e3e03486b116af23da69f1dade6be96c/68747470733a2f2f692e696d6775722e636f6d2f774d32433341612e706e67)
***
![coordinate](https://camo.githubusercontent.com/fb2fef71e34d9db89140613202e0b57954d4cc63/68747470733a2f2f692e696d6775722e636f6d2f384473695835312e706e67)
***
![contour](https://camo.githubusercontent.com/debbbccaab8b714aea3789bddf3c15750098a13c/68747470733a2f2f692e696d6775722e636f6d2f4b533861704f652e706e67)

## B. Setup

### Install on virtual environment on Windows 10
See [page](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Windows-10-setup) from wiki.

### General setup guide

#### Required
* Install python 3.8 or higher
  * Visit https://www.python.org/downloads/
* Install optuna
  * pip install optuna
  
#### Visualization
* pip install plotly
* pip install scikit-learn==0.24.2
* pip install kaleido

#### Save studies to pandas dataframe and csv file
* pip install pandas

#### To use skopt sampler
* pip install scikit-optimize

#### To use botorch
* pip install botorch

#### Install all dependencies
Instead of installing each module like optuna, plotly and others. Just install with requirements.txt.  
* pip install -r requirements.txt
  
## C. Basic optimization process outline
1. Prepare the engines and the parameters to be optimized. Set max_trial to 1000 or so.
2. Setup a game match between 2 engines, the test engine and base engine. The test engine will use the parameters suggested by optuna optimizer while the base engine will use the initial or default parameter values. In the beginning the best parameter is the initial parameter. The test engine will use the initial parameter values suggested by the optimimzer.
3. After a match of say 24 games, the result of test engine will be sent to the optimizer. The result can be a score rate like (wins+draw/2)/24 or an Elo from the point of view of the test engine. If the test engine wins (score > 0.5) or elo > 0.0, update the best parameter to the parameter used by test engine.
4. Check the number of trials done. If trial >= max_trial stop the optimization and save the plots. This is automatically done by the optimizer.
5. Get the new parameter values suggested by the optimizer. This will be used by the test engine.
6. Run a new match, goto step 3.
7. You can extend the study or optimization by running the study again using the same study_name and conditions.

## D. Supported Samplers/Optimizers
* [TPE](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler) or Tree-structured Parzen Estimator
* [BOTorch](https://github.com/pytorch/botorch) or Bayesian Optimization in PyTorch.
* [CMAES](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.PyCmaSampler.html) or Covariance Matrix Adaptation Evolution Strategy
* [SKOPT](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.SkoptSampler.html) or [scikit-optimize](https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html#skopt.optimizer.Optimizer)
  * acquisition_function
    * LCB
      * kappa=1.96, default
      * kappa=10000, explore
      * kappa=0.0001, exploit
    * EI
      * xi = 0.01, default
      * xi = 10000, explore
      * xi = 0.0001, exploit
    * PI
      * xi = 0.01, default
      * xi = 10000, explore
      * xi = 0.0001, exploit
    * gp_hedge (default)
  * base_estimator
    * GP - Gaussian Process
    * RF - Random Forest
    * ET - Extra Tree
    * GBRT - Gradient Boosted Regression Trees
  * acq_optimizer
    * auto
    * sampling
    * lbfgs

## E. Help
See [help](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Help) in wiki.
Or type `python tuner.py -h`

## F. Sample command line
#### Using tpe optimizer
```python
python tuner.py --sampler name=tpe --engine ./engines/deuterium/deuterium --concurrency 6 --opening-file ./start_opening/ogpt_chess_startpos.epd --opening-format epd --input-param "{'PawnValueEn': {'default':92, 'min':90, 'max':120, 'step':2}, 'BishopValueOp': {'default':350, 'min':290, 'max':350, 'step':3}}" --games-per-trial 24 --plot --base-time-sec 15 --inc-time-sec 0.1 --study-name study1 --pgn-output study1.pgn --trials 100 --common-param "{'Hash': 128}"
```

#### Use Elo as objective value instead of score rate
Use the flag  
`--elo-objective`

#### Deterministic and Non-Deterministic objective function
Our objective function result is the result of engine vs engine match. There are engines at fixed depth move control that are deterministic that is if you play the same opening at fixed depth of 2 for 100 games and repeat the same the result of the match is the same. The samplers such as TPE, CMAES, SKOPT and BOTorch may suggest parameter values that were already suggested before. By default the tuner will not replay the match it will just return the previous result.

There is a flag that play a match for repeated parameter suggestions and it is called `--noisy-result`. This is mainly applied when more than one same parameter matches produces different results this is called non-determinisitic or stochastic result. An example situation is when you play a match with a time control instead of fixed depth. Conduct a match #1 at time control of 5s+100ms for 100 games with opening set #1, then do match #2 with opening set #2, most likely the result is not the same. Note that during matches each opening is played twice. In this case it is better to add the `--noisy-result` flag in the command line.

An example log when `--noisy-result` flag is enabled and sampler repeats suggesting param values. Objective value type is elo with `--elo-objective` flag.  
```python
starting trial: 149 ...
deterministic function: False
Duplicate suggestion from sampler, {'Pp2': 10, 'Pp6': 3}
Execute engine match as --noisy-result flag is enabled.
suggested param for test engine: {'Pp2': 10, 'Pp6': 3}
param for base engine          : {'Pp2': 7, 'Pp6': 2}
common param: {'Hash': 128, 'EvalHash': 4}
init param: {'Pp2': 7, 'Pp6': 2}
init objective value: 0.0
study best param: {'Pp2': 10, 'Pp6': 1}
study best objective value: Elo 124.0
study best trial number: 1
Actual match result: Elo 22.0, CI: [-75.9, +119.4], CL: 95%, G/W/D/L: 32/11/12/9, POV: optimizer
Elo Diff: +21.7, ErrMargin: +/- 97.6, CI: [-75.9, +119.4], LOS: 67.3%, DrawRatio: 37.50%
test param format for match manager: option.Pp2=10 option.Pp6=3
result sent to optimizer: 22.0
elapse: 0h:0m:19s
Trial 149 finished with value: 22.0 and parameters: {'Pp2': 10, 'Pp6': 3}. Best is trial 1 with value: 124.0.
```

#### Command line with float parameter values
Add a key value pair of `'type': 'float'`
```python
--input-param "{'CPuct': {'default':2.147, 'min':1.0, 'max':3.0, 'step':0.05, 'type': 'float'}, 'CPuctBase': {'default':18368.0, 'min':15000.0, 'max':20000.0, 'step':2.0, 'type': 'float'}, 'CPuctFactor': {'default':2.82, 'min':0.5, 'max':3.5, 'step':0.05, 'type': 'float'}, 'FpuValue': {'default':0.443, 'min':-0.1, 'max':1.2, 'step':0.05, 'type': 'float'}, 'PolicyTemperature': {'default':1.61, 'min':0.5, 'max':3.0, 'step':0.05, 'type': 'float'}}"
```


## G. Optimization studies

* [Chess Piece Value Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization)
* [Chess Evaluation Positional Parameter Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-Evaluation-Positional-Parameter-Optimization)
* [Search Parameter Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Search-Parameter-Optimization)
* [Optimization with Threshold Pruner](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/commit/eb595ecb7a752cf2db6d8752b7480c59f696c7b7#commitcomment-42769655)
* [Optimization Performance Comparison](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Performance-comparison)
* [Performance comparison between tpe multivariate and cmaes](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Performance-comparison-between-tpe-multivariate-and-cmaes)

## H. Credits
* Optuna  
https://github.com/optuna/optuna
* Cutechess  
https://github.com/cutechess/cutechess
* Plotly  
https://plotly.com/
* Sklearn  
https://scikit-learn.org/stable/
* [Stockfish](https://stockfishchess.org/)
