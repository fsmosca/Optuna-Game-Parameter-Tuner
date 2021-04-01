# Optuna Game Parameter Tuner
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
* pip install sklearn
* pip install kaleido

#### Save studies to pandas dataframe and csv file
* pip install pandas

#### To use skopt sampler
* pip install scikit-optimize

#### Install all dependencies
Instead of installing each module like optuna, plotly and others. Just install with requirements.txt.  
* pip install -r requirements.txt
  
## C. Basic optimization process outline
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

## D. Supported optimizers
* [TPE](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler) or Tree-structured Parzen Estimator
* [CMAES](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler) or Covariance Matrix Adaptation Evolution Strategy
* [skopt](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.SkoptSampler.html) or [scikit-optimize](https://scikit-optimize.github.io/stable/modules/generated/skopt.optimizer.Optimizer.html#skopt.optimizer.Optimizer)
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
    * gp_hedge
  * base_estimator
    * GP - Gaussian Process
    * RF - Random Forest
    * ET - Extra Tree
    * GBRT - Gradient Boosted Regression Trees

## E. Help
See [help](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Help) in wiki.

## F. Sample command line
#### Using tpe optimizer
```python
python tuner.py --sampler name=tpe --engine ./engines/deuterium/deuterium --concurrency 6 --opening-file ./start_opening/ogpt_chess_startpos.epd --opening-format epd --input-param "{'PawnValueEn': {'default':92, 'min':90, 'max':120, 'step':2}, 'BishopValueOp': {'default':350, 'min':290, 'max':350, 'step':3}}" --initial-best-value 0.54 --games-per-trial 200 --plot --base-time-sec 120 --inc-time-sec 0.1 --study-name study1 --pgn-output study1.pgn --trials 200 --common-param "{'Hash': 128}"
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
