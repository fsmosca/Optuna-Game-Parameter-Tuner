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
OS: Windows 10, python 3.8
The engine starts with piece values set at 0. Can the optimizer improve the initial values? Let's see the results after 1000 trials. Pawn values should be close to 100, knight values close to 300, bishop values close to 300, rook values close to 500 and queen values close to 1000. 

```python
python tuner.py --engine ./engines/deuterium/deuterium.exe --opening-file ./start_opening/ogpt_chess_startpos.epd --concurrency 6 --games-per-trial 12 --base-time-sec 2
trials: 1000, games_per_trial: 12
[I 2020-09-10 17:59:56,590] A new study created in RDB with name: piece_value
Warning, best param from previous trial is not found!, Use the init param.
init param: {'PawnValueOp': 0, 'PawnValueEn': 0, 'KnightValueOp': 0, 'KnightValueEn': 0, 'BishopValueOp': 0, 'BishopValueEn': 0, 'RookValueOp': 0, 'RookValueEn': 0, 'QueenValueOp': 0, 'QueenValueEn': 0}
Warning, init value is not found!
init best value: 0.5
starting trial: 0 ...
suggested param: {'PawnValueOp': 466, 'PawnValueEn': 381, 'KnightValueOp': 175, 'KnightValueEn': 512, 'BishopValueOp': 36, 'BishopValueEn': 431, 'RookValueOp': 277, 'RookValueEn': 677, 'QueenValueOp': 383, 'QueenValueEn': 1995}
init param: {'PawnValueOp': 0, 'PawnValueEn': 0, 'KnightValueOp': 0, 'KnightValueEn': 0, 'BishopValueOp': 0, 'BishopValueEn': 0, 'RookValueOp': 0, 'RookValueEn': 0, 'QueenValueOp': 0, 'QueenValueEn': 0}
init value: 0.5
[I 2020-09-10 18:00:07,516] Trial 0 finished with value: 0.5005 and parameters: {'PawnValueOp': 466, 'PawnValueEn': 381, 'KnightValueOp': 175, 'KnightValueEn': 512, 'BishopValueOp': 36, 'BishopValueEn': 431, 'RookValueOp': 277, 'RookValueEn': 677, 'QueenValueOp': 383, 'QueenValueEn': 1995}. Best is trial 0 with value: 0.5005.
starting trial: 1 ...
suggested param: {'PawnValueOp': 750, 'PawnValueEn': 403, 'KnightValueOp': 106, 'KnightValueEn': 306, 'BishopValueOp': 418, 'BishopValueEn': 129, 'RookValueOp': 112, 'RookValueEn': 681, 'QueenValueOp': 1840, 'QueenValueEn': 586}
best param: {'PawnValueOp': 466, 'PawnValueEn': 381, 'KnightValueOp': 175, 'KnightValueEn': 512, 'BishopValueOp': 36, 'BishopValueEn': 431, 'RookValueOp': 277, 'RookValueEn': 677, 'QueenValueOp': 383, 'QueenValueEn': 1995}
best value: 0.5005
[I 2020-09-10 18:00:23,233] Trial 1 finished with value: 0.0 and parameters: {'PawnValueOp': 750, 'PawnValueEn': 403, 'KnightValueOp': 106, 'KnightValueEn': 306, 'BishopValueOp': 418, 'BishopValueEn': 129, 'RookValueOp': 112, 'RookValueEn': 681, 'QueenValueOp': 1840, 'QueenValueEn': 586}. Best is trial 0 with value: 0.5005.
starting trial: 2 ...
suggested param: {'PawnValueOp': 155, 'PawnValueEn': 97, 'KnightValueOp': 10, 'KnightValueEn': 644, 'BishopValueOp': 934, 'BishopValueEn': 916, 'RookValueOp': 299, 'RookValueEn': 44, 'QueenValueOp': 745, 'QueenValueEn': 955}
best param: {'PawnValueOp': 466, 'PawnValueEn': 381, 'KnightValueOp': 175, 'KnightValueEn': 512, 'BishopValueOp': 36, 'BishopValueEn': 431, 'RookValueOp': 277, 'RookValueEn': 677, 'QueenValueOp': 383, 'QueenValueEn': 1995}
best value: 0.5005

...

[I 2020-09-10 18:01:41,567] Trial 5 finished with value: 0.5018329999999999 and parameters: {'PawnValueOp': 120, 'PawnValueEn': 395, 'KnightValueOp': 279, 'KnightValueEn': 657, 'BishopValueOp': 427, 'BishopValueEn': 608, 'RookValueOp': 886, 'RookValueEn': 759, 'QueenValueOp': 906, 'QueenValueEn': 1934}. Best is trial 5 with value: 0.5018329999999999.
starting trial: 6 ...
suggested param: {'PawnValueOp': 5, 'PawnValueEn': 31, 'KnightValueOp': 876, 'KnightValueEn': 122, 'BishopValueOp': 812, 'BishopValueEn': 34, 'RookValueOp': 204, 'RookValueEn': 17, 'QueenValueOp': 258, 'QueenValueEn': 546}
best param: {'PawnValueOp': 120, 'PawnValueEn': 395, 'KnightValueOp': 279, 'KnightValueEn': 657, 'BishopValueOp': 427, 'BishopValueEn': 608, 'RookValueOp': 886, 'RookValueEn': 759, 'QueenValueOp': 906, 'QueenValueEn': 1934}
best value: 0.5018329999999999

...
```

## Credits
* Optuna  
https://github.com/optuna/optuna
