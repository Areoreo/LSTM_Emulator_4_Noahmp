


python 06_calibration_applying_emulator.py     --forcing data/raw/forcing/forcing_sample_1.nc     --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv     --bounds value_bounds.csv     --num_calibration 200     --max_iter 500     --lr 0.01    --model_dir results_forward_comprehensive/AttentionLSTM_20251120_100834_dim-1024_layer-4

python 06_calibration_applying_emulator_multiple_runs.py     --forcing data/raw/forcing/forcing_sample_1.nc     --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv     --bounds value_bounds.csv     --num_calibration 10     --max_iter 500    --model_dir results_forward_comprehensive/AttentionLSTM_20251120_100834_dim-1024_layer-4