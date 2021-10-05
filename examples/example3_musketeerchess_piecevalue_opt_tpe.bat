:: This batch file will try to optimize the hawk/unicorn piece values in the musketeer-stockfish-modern.exe.
:: Default values:
::   {'HawkValueEg': 1561, 'HawkValueMg': 1537, 'UnicornValueEg': 1772, 'UnicornValueMg': 1584}


:: Copy this file in the folder where tuner.py is located. Modify the paths in --engine and --opening-file if necessary.
::   Be sure to activate your virtual environment so that the modules used by tuner.py will be imported without issues.


:: --sampler name=tpe
::   This optimization uses the tpe sampler. Optuna has some samplers to use.
::   https://github.com/fsmosca/Optuna-Game-Parameter-Tuner#d-supported-optimizers
::   type tuner.py --help to see samplers supported by this tuner.
::   Examples:
::   --sampler name=cmaes
::   or
::   --sampler name=skopt


:: --games-per-trial
::   The number of games to be played to determine the objective value. More games is better, typically 500 or more.


:: --trials
::   Number of trials to be performed. If there are more parameters and wide range of parameter values, trials
::   should be increased. When number of trials are completed, you can extend the study by runnning again the same
::   batch file.


:: --threshold-pruner result=0.45
::   The GAMESPERTRIAL in this example is 300, the tuner will run the match at 150 games first followed by another 150 games.
::   After the first 150 games are finished and the testengine result is below --threshold-pruner result value then the tuner
::   will not continue with the second match. The tuner will inform the optimizer that this trial is pruned. This will save
::   time in our simulation.


:: --concurrency value
::   During the engine vs engine match, if concurrency value is 4, then there will be 4 games that will be run concurrently or in parallel.
::   If your processor has more threads of say 16, you may use 12 to complete the match faster.


:: --base-time-sec
::   The base time cotrol used during engine vs engine match. It is better to use a high number of base time especially
::   when the parameter to be optimized is a search parameter.


:: --plot
::   After every 10 trials are completed, plots will be generated, and can be found in visuals folder.
::   Filenames will be <study_name>_<trial_number><plot_type>.png
::   plot_type are importance, parallel, contour and slice and hist


:: It is better to use high GAMESPERTRIAL value to lower the uncentainty of the match result. A high value of --initial-best-value
::   also helps in finding the best param with high centainty and also eliminates the noise in engine vs engine match.


:: When the optimizer could not suggest a better param values meaning that it could not exceed the --initial-best-value value then
::   the best param is still the init param. Or It can happen that the testengine will defeat the baseengine by a result of only 0.52, that
::   is still more than 50%, so the param of that trial can be used against the default param in the verification test.


:: Interrupt and resume
::   You can interrupt the optimization and continue later, trial histories are saved.


:: Verification test
::   The resulting optimized values can then be tested in more than GAMESPERTRIAL against the default or init param.


:: Modify options here
set COMBO=hawk_unicorn
set CONCURRENCY=1
set GAMESPERTRIAL=24
set NUMTRIALS=100
set SAMPLER=tpe


python -u tuner.py --study-name musketeer_%COMBO%_piecevalues_tpe ^
--sampler name=%SAMPLER% ^
--games-per-trial %GAMESPERTRIAL% --trials %NUMTRIALS% ^
--concurrency %CONCURRENCY% ^
--base-time-sec 2 ^
--inc-time-sec 0.05 ^
--draw-movenumber 30 --draw-movecount 6 --draw-score 0 ^
--resign-movecount 3 --resign-score 500 ^
--engine ./engines/musketeer/musketeer-stockfish/musketeer-stockfish-modern.exe ^
--input-param "{'HawkValueMg': {'default':1537, 'min':1437, 'max':1637, 'step':2}, 'HawkValueEg': {'default':1561, 'min':1461, 'max':1661, 'step':2}, 'UnicornValueMg': {'default':1584, 'min':1484, 'max':1684, 'step':2}, 'UnicornValueEg': {'default':1772, 'min':1672, 'max':1872, 'step':2}}" ^
--opening-file ./start_opening/musketeer/hawk_unicorn_startpos.fen ^
--variant musketeer --match-manager duel --threshold-pruner result=0.45 ^
--plot

