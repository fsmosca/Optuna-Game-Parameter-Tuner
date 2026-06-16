REM Run a whole tuning session from a single unified config file.
REM Edit yaml_files\deuterium_config.yaml to set input_param, common_param and options.
REM Any flag added below overrides the matching value in the config file,
REM e.g. here we override trials and games-per-trial for a quick test run.

python tuner.py --config yaml_files/deuterium_config.yaml --trials 20 --games-per-trial 20
