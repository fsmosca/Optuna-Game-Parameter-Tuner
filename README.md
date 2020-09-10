# Optuna Game Parameter Tuner
A game search and evaluation parameter tuner using optuna framework. The game can be a chess or other game variants.

## Setup

* Install python 3.8 or higher
* Install optuna
  * pip install optuna

## Help
```python
python tuner.py -h
usage: tuner.py [-h] --engine ENGINE [--hash HASH] [--trials TRIALS] [--concurrency CONCURRENCY]
                [--games-per-trial GAMES_PER_TRIAL] [--base-time-sec BASE_TIME_SEC] [--inc-time-sec INC_TIME_SEC]
                --opening-file OPENING_FILE [--variant VARIANT] [--pgn-output PGN_OUTPUT]

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

#### Chess Piece value optimization
The piece values from pawn to queen were [optimized](https://github.com/fsmosca/Optuna-Game-Parameter-Tuner/wiki/Chess-piece-value-optimization) from 0 values.

## Credits
* Optuna  
https://github.com/optuna/optuna
