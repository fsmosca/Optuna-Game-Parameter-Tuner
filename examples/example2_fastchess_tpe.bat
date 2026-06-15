python tuner.py --study-name sf_d1_fastchess_tpe --match-manager fastchess --match-manager-file "C:\chess\fastchess-windows-latest\fastchess.exe" ^
--sampler name=tpe --engine ./engines/stockfish-modern/stockfish.exe --concurrency 4 --opening-file ./start_opening/ogpt_chess_startpos.epd --opening-format epd ^
--games-per-trial 20 --trials 20 --plot --depth 1 --pgn-output sf_d1_fastchess_tpe.pgn --threshold-pruner result=0.35 --use-affinity 0-4 ^
--input-param "{'eMobilityBonus[2][10]': {'default':158, 'min':100, 'max':200, 'step':2}, 'mOutpost[0]': {'default':56, 'min':0, 'max':100, 'step':4}, 'eThreatByMinor[4]': {'default':119, 'min':50, 'max':150, 'step':1}, 'mThreatBySafePawn': {'default':173, 'min':120, 'max':220, 'step':1}}"
