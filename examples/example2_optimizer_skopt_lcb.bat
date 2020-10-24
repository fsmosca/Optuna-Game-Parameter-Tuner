:: Example 2

:: --sampler name=skopt acquisition_function=LCB

:: --concurrency 1
:: Increase this value if your computer has more threads or cores.



python tuner.py --study-name sf_d1_skopt_LCB --sampler name=skopt acquisition_function=LCB --fix-base-param --engine ./engines/stockfish-modern/stockfish.exe --hash 64 --concurrency 1 --opening-file ./start_opening/ogpt_chess_startpos.epd --input-param "{'eMobilityBonus[2][10]': {'default':158, 'min':100, 'max':200, 'step':4}, 'mOutpost[0]': {'default':56, 'min':0, 'max':100, 'step':4}, 'eThreatByMinor[4]': {'default':119, 'min':50, 'max':150, 'step':4}, 'mThreatBySafePawn': {'default':173, 'min':120, 'max':220, 'step':4}}" --games-per-trial 1000 --trials 100 --plot --base-time-sec 60 --inc-time-sec 0.1 --depth 1 --pgn-output train_sf_d1.pgn --threshold-pruner result=0.35
