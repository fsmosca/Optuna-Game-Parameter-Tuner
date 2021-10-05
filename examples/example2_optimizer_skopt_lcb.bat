:: Example 2
:: Usage, if you want to use this bat file, copy this file to the location of tuner.py.

:: --sampler name=skopt acquisition_function=LCB

:: --concurrency 1
:: Increase this value if your computer has more threads or cores.



python tuner.py --study-name sf_d1_skopt_LCB --sampler name=skopt acquisition_function=LCB --engine ./engines/stockfish-modern/stockfish.exe --concurrency 1 --opening-file ./start_opening/ogpt_chess_startpos.epd --opening-format epd --input-param "{'eMobilityBonus[2][10]': {'default':158, 'min':100, 'max':200, 'step':2}, 'mOutpost[0]': {'default':56, 'min':0, 'max':100, 'step':4}, 'eThreatByMinor[4]': {'default':119, 'min':50, 'max':150, 'step':1}, 'mThreatBySafePawn': {'default':173, 'min':120, 'max':220, 'step':1}}" --games-per-trial 1000 --trials 100 --plot --depth 1 --pgn-output train_sf_d1_skopt_LCB.pgn --threshold-pruner result=0.35
