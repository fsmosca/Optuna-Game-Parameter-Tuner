# Optuna Game Parameter Tuner
A game search and evaluation parameter tuner using optuna framework. The game can be a chess or other game variants. Engine evaluation parameters that can be optimized are piece values like pawn value or knight value and others. Search parameters that can be optimized are futility pruning margin, null move reduction factors and others. 

## Setup

#### Required
* Install python 3.8 or higher
* Install optuna
  * pip install optuna
  
#### Visualization
* pip install plotly
* pip install kaleido
  
## Basic optimization process outline
1. Prepare the engines and the parameters to be optimized. Set max_trial to 1000 or so.
2. Setup a game match between 2 engines. Test engine and base engine. Test engine will use the parameters suggested by optuna optimizer while base engine will use the initial default parameter values. In the beginning the best parameter is the initial parameters. The test will get the initial parameter values suggestion from the optimimzer.
3. After a match of say 24 games, the score of test engine will be sent to the optimizer. The score is just (wins+draw/2)/24 from the point of view of the test engine. If the test engine wins (score > 0.5), update the best parameter to the parameter used by test engine. Increment trial by 1.
4. Check the number of trials done. If trial >= max_trial stop the optimization. This is done by the optimizer.
5. Get the new parameter values suggested by the optimizer.
6. Run a new match, the base engine will use the best parameter while the test engine will use the new parameter values suggested by the optimizer.
7. Goto step 3.

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


## Test

* [Chess Piece value optimization 1](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization)

## Credits
* Optuna  
https://github.com/optuna/optuna
