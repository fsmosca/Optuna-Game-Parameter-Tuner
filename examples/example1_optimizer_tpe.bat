:: Example 1
:: Usage, if you want to use this bat file, copy this file to the location of tuner.py.


:: --study-name sf_d1
:: The name of study. It is recomended to define a name as this
:: name will be used to continue the study after an interruption,
:: is also used in the filename of image file from a plot, and
:: is also used to save the optimization data in csv file.

:: --sampler name=tpe
:: The optimizer is TPE.

:: --fix-base-param
:: The best param is the default param and is always used by the base_engine.
:: test_engine vs base_engine, and test_engine will use param values from optimizer.

:: --engine ./engines/stockfish-modern/stockfish.exe
:: The path and filename of engine.

:: --concurrency 2
:: During the engine match, games will be played in parallel or concurrently by this number.
:: If your PC has more threads or cores, you can use it to finish the optimization faster.

:: --input-param "{'eMobilityBonus[2][10]': {'default':158, 'min':100, 'max':200, 'step':4}, 'mOutpost[0]': {'default':56, 'min':0, 'max':100, 'step':4}, 'eThreatByMinor[4]': {'default':119, 'min':50, 'max':150, 'step':4}, 'mThreatBySafePawn': {'default':173, 'min':120, 'max':220, 'step':4}}"
:: The parameters to be optimized.

:: --games-per-trial 1000
:: Number of games to play to get the objective or loss value in every trial.

:: --trials 200
:: Number of trials to execute.

:: --base-time-sec 60
:: When using time control.

:: --inc-time-sec 0.1
:: When using time control with increment.

:: --depth 1
:: Games in the match will be played at this depth value.
:: See the above options with time control.

:: --plot
:: After the study a plot will be saved in visuals folder.

:: --threshold_pruner result=0.35
:: After 500 games, if the partial result of the match is 0.30 from the point of view of the optimizer param values
:: then this trial will be stopped and new trial will be started. However if result is 0.35 and above then
:: the match will continue for another 500 games to complete the games per trial requirement of 1000.


python tuner.py --study-name sf_d1 --sampler name=tpe --fix-base-param --engine ./engines/stockfish-modern/stockfish.exe --hash 64 --concurrency 6 --opening-file ./start_opening/ogpt_chess_startpos.epd --opening-format epd --input-param "{'eMobilityBonus[2][10]': {'default':158, 'min':100, 'max':200, 'step':4}, 'mOutpost[0]': {'default':56, 'min':0, 'max':100, 'step':4}, 'eThreatByMinor[4]': {'default':119, 'min':50, 'max':150, 'step':4}, 'mThreatBySafePawn': {'default':173, 'min':120, 'max':220, 'step':4}}" --games-per-trial 1000 --trials 200 --plot --base-time-sec 60 --inc-time-sec 0.1 --depth 1 --pgn-output train_sf_d1.pgn --threshold-pruner result=0.35
