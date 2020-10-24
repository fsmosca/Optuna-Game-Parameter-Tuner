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

### Install on virtual environment on Windows 10
* Install python 3.8 or later.  
  * Link: https://www.python.org/downloads/
* Create game_param_tuner folder on your c or other drive. I will use my d drive. Use your windows explorer to create a folder. It would look like this.  
  `d:\game_param_tuner`
* Download this repo, see at the top right.  
  * Code->Download ZIP
* Put the downloaded file `Optuna-Game-Parameter-Tuner-master.zip` into the `game_param_tuner` folder.  
* Run powershell as administrator.  
  * In the search box at lower left of window, type `powershell` and select `Run as administrator`. You should see this.  
  `PS C:\WINDOWS\system32>`  
* Change to `game_param_tuner` folder.  
  `PS C:\WINDOWS\system32> cd d:\game_param_tuner`  
* Check the contents of the current folder by typing dir.   
  `PS D:\game_param_tuner> dir`  
  You should see this.  
```
      Directory: D:\game_param_tuner


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        12/10/2020  11:30 pm       16287737 Optuna-Game-Parameter-Tuner-master.zip
```
* Unzip the file  
  `PS D:\game_param_tuner> Expand-Archive Optuna-Game-Parameter-Tuner-master.zip .\`  
* Type dir to see the folder `Optuna-Game-Parameter-Tuner-master`  
  `PS D:\game_param_tuner> dir`  
  You should see this.  
```
      Directory: D:\game_param_tuner


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        13/10/2020   2:05 pm                Optuna-Game-Parameter-Tuner-master
-a----        12/10/2020  11:30 pm       16287737 Optuna-Game-Parameter-Tuner-master.zip
```
* Change to folder `Optuna-Game-Parameter-Tuner-master`.  
  `PS D:\game_param_tuner> cd Optuna-Game-Parameter-Tuner-master`  
* See the contents of current folder.  
  `PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> dir`  
  You should see this.  
```
      Directory: D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        13/10/2020   2:21 pm                engines
d-----        13/10/2020   2:21 pm                start_opening
d-----        13/10/2020   2:21 pm                tools
d-----        13/10/2020   2:21 pm                tourney_manager
d-----        13/10/2020   2:21 pm                visuals
-a----        12/10/2020   5:33 am           1865 .gitignore
-a----        12/10/2020   5:33 am           1064 LICENSE
-a----        12/10/2020   5:33 am          11202 README.md
-a----        12/10/2020   5:33 am            636 requirements.txt
-a----        12/10/2020   5:33 am          32886 tuner.py
```

* Check the version of your installed python.  
  `PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> python --version`  
  This is what I have.  
  `Python 3.8.5`  
  
* Create virtual environment on myvenv.  
  `PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> python -m venv myvenv`  
  
* Type dir and notice the folder myvenv.  
  `PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> dir`  
```
    Directory: D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        13/10/2020   2:21 pm                engines
d-----        13/10/2020   2:33 pm                myvenv
d-----        13/10/2020   2:21 pm                start_opening
d-----        13/10/2020   2:21 pm                tools
d-----        13/10/2020   2:21 pm                tourney_manager
d-----        13/10/2020   2:21 pm                visuals
-a----        12/10/2020   5:33 am           1865 .gitignore
-a----        12/10/2020   5:33 am           1064 LICENSE
-a----        12/10/2020   5:33 am          11202 README.md
-a----        12/10/2020   5:33 am            636 requirements.txt
-a----        12/10/2020   5:33 am          32886 tuner.py
```
  
* Activate the virtual environment.  
  * Modify the restriction first.  
    `PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> Set-ExecutionPolicy unrestricted`  
    Then type A  
  * Activate it.  
  `PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> .\myvenv\scripts\activate`  
  The prompt changes and would look like this.  
  `(myvenv) PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master>`
  
* Install the requirements.  
  `(myvenv) PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> pip install -r requirements.txt`  
  Wait for it to finish.  
  `(myvenv) PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master>`
  
This setup is done. You can now optimize a param of an engine. Lets try to optimize 2 search param of Stockfish. The `FutMargin` and `RazorMargin`.  

#### Command line:  
```
(myvenv) PS D:\game_param_tuner\Optuna-Game-Parameter-Tuner-master> python tuner.py --engine ./engines/stockfish-modern/stockfish.exe --hash 64 --concurrency 1 --opening-file ./start_opening/ogpt_chess_startpos.epd --input-param "{'RazorMargin': {'default':527, 'min':250, 'max':650, 'step':4}, 'FutMargin': {'default':227, 'min':50, 'max':350, 'step':4}}" --plot --base-time-sec 120 --depth 4 --study-name sample --pgn-output sample.pgn --trials 100 --games-per-trial 20 --sampler name=tpe
```  

You should see something like this.  
```
2020-10-13 15:14:08,054 | INFO  | trials: 100, games_per_trial: 20, sampler: [['name=tpe']]

2020-10-13 15:14:08,054 | INFO  | input param: OrderedDict([('FutMargin', {'default': 227, 'min': 50, 'max': 350, 'step': 4}), ('RazorMargin', {'default': 527, 'min': 250, 'max': 650, 'step': 4})])

2020-10-13 15:14:08,054 | INFO  | Starting optimization ...
2020-10-13 15:14:10,087 | INFO  | A new study created in RDB with name: sample
2020-10-13 15:14:10,244 | WARNI | Warning, best value from previous trial is not found!
2020-10-13 15:14:10,244 | INFO  | study best value: 0.0
2020-10-13 15:14:10,244 | WARNI | Warning, best param from previous trial is not found!.
2020-10-13 15:14:10,244 | INFO  | study best param: {}
2020-10-13 15:14:10,432 | INFO  |
2020-10-13 15:14:10,432 | INFO  | starting trial: 0 ...
2020-10-13 15:14:10,744 | INFO  | suggested param for test engine: {'FutMargin': 110, 'RazorMargin': 634}
2020-10-13 15:14:10,744 | INFO  | param for base engine          : {'FutMargin': 227, 'RazorMargin': 527}

2020-10-13 15:14:10,744 | INFO  | init param: {'FutMargin': 227, 'RazorMargin': 527}
2020-10-13 15:14:10,760 | INFO  | init value: 0.5
2020-10-13 15:14:10,760 | INFO  | study best param: {}
2020-10-13 15:14:10,760 | INFO  | study best value: 0.0

2020-10-13 15:14:13,309 | INFO  | Actual match result: 0.45, point of view: optimizer suggested values
2020-10-13 15:14:13,481 | INFO  | Trial 0 finished with value: 0.44 and parameters: {'FutMargin': 110, 'RazorMargin': 634}. Best is trial 0 with value: 0.44.

...
```

#### Important options to observe
* `--engine ./engines/stockfish-modern/stockfish.exe`, this is the location of the engine whose parameters will be optimized.  
* `--games-per-trial 20`, number of games to play to generate an objective value.  
* `--trials 100`, total trials to execute, the same number of objective values will be generated.  
* `--concurrency 1`, the tuner will use 1 thread from your processor to play a game in a match of 20 games. If your processor has more threads of say 4, you can use 2 as concurrency value to make the optimization faster.

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
See [help](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Help) in wiki.

## Sample command line
```python
python tuner.py --engine ./engines/deuterium/deuterium --hash 128 --concurrency 6 --opening-file ./start_opening/ogpt_chess_startpos.epd --input-param "{'PawnValueEn': {'default':92, 'min':90, 'max':120, 'step':2}, 'BishopValueOp': {'default':350, 'min':290, 'max':350, 'step':3}, 'BishopValueEn': {'default':350, 'min':290, 'max':350, 'step':3}, 'RookValueEn': {'default':525, 'min':480, 'max':550, 'step':5}, 'QueenValueOp': {'default':985, 'min':950, 'max':1200, 'step':5}}" --initial-best-value 0.54 --games-per-trial 200 --plot --base-time-sec 120 --inc-time-sec 0.1 --depth 4 --study-name pv_d4_eisample_50_pruner --pgn-output train_pv_d4_eisamples_50_pruner.pgn --trials 200 --threshold-pruner result=0.35 --sampler name=skopt acquisition_function=LCB
```


## Optimization studies

* [Chess Piece Value Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization)
* [Chess Evaluation Positional Parameter Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-Evaluation-Positional-Parameter-Optimization)
* [Search Parameter Optimization](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Search-Parameter-Optimization)
* [Optimization with Threshold Pruner](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/commit/eb595ecb7a752cf2db6d8752b7480c59f696c7b7#commitcomment-42769655)
* [Optimization Performance Comparison](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Performance-comparison)

## Credits
* Optuna  
https://github.com/optuna/optuna
* Cutechess  
https://github.com/cutechess/cutechess
* Plotly  
https://plotly.com/
* Sklearn  
https://scikit-learn.org/stable/
* [Stockfish](https://stockfishchess.org/)
