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
  
## Basic optimization process outline
1. Prepare the engines and the parameters to be optimized. Set max_trial to 1000 or so.
2. Setup a game match between 2 engines. Test engine and base engine. Test engine will use the parameters suggested by optuna optimizer while base engine will use the initial default parameter values. In the beginning the best parameter is the initial parameter. The test engine will use the initial parameter values suggested by the optimimzer.
3. After a match of say 24 games, the score of test engine will be sent to the optimizer. The score is just (wins+draw/2)/24 from the point of view of the test engine. If the test engine wins (score > 0.5), update the best parameter to the parameter used by test engine. Increment trial by 1.
4. Check the number of trials done. If trial >= max_trial stop the optimization. This is done by the optimizer.
5. Get the new parameter values suggested by the optimizer. This will be used by the test engine.
6. Run a new match, the base engine will use the best parameter while the test engine will use the new parameter values suggested by the optimizer.
7. Goto step 3.
8. When max_trial is reached, optimization is stopped and png plots will be saved.

## Help
```python
python tuner.py -h
usage: tuner.py [-h] --engine ENGINE [--hash HASH] [--trials TRIALS] [--concurrency CONCURRENCY]
                [--games-per-trial GAMES_PER_TRIAL] [--study-name STUDY_NAME] [--base-time-sec BASE_TIME_SEC]
                [--inc-time-sec INC_TIME_SEC] --opening-file OPENING_FILE [--variant VARIANT] [--pgn-output PGN_OUTPUT]

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
  --opening-file OPENING_FILE
                        Start opening filename in fen or epd format.
  --variant VARIANT     Game variant, default=normal.
  --pgn-output PGN_OUTPUT
                        Output pgn filename, default=optuna_games.pgn.
```

## Command line
```python
python tuner.py --engine ./engines/deuterium/deuterium.exe --opening-file ./start_opening/ogpt_chess_startpos.epd
```


## Tests

* Optimization 1
  * Link: [Chess Piece value optimization 1](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization)

* Optimization 2
  * Trials: 250
  * Games per Trial: 32
  * TC: 2s+100ms
  * Link: [Chess Piece value optimization 2](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization-2)
  
* Optimization 3
  * Trials: 212
  * Games per trial: 24
  * TC: 10s+100ms
  * Link: [Chess Piece value optimization 3](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization-3)

## Credits
* Optuna  
https://github.com/optuna/optuna
* Plotly  
https://plotly.com/
* Sklearn  
https://scikit-learn.org/stable/
