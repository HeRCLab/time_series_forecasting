#!/bin/bash
date
#export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-12.2/bin:/share/iahmad/anaconda3/bin:$PATH"

#CUDA_VISIBLE_DEVICES="0" python3 transformer_tutorial_v_wfh.py
#python -c 'import torch as tc; print(tc.__version__)'
#nvcc -V
#kill $(nvidia-smi | awk '$1=="Processes:" {p=1} p && $1 == 1 && $3 > 0 {print $3}')
#lsof /dev/nvidia* | awk '{print $1}' | xargs -I {} kill {}
export 'PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:32'
#CUDA_VISIBLE_DEVICES="0" python3 transformer_tutorial_ifty_v_memory_issue.py --n_samples 1
#CUDA_LAUNCH_BLOCKING=1

pip install .

CUDA_VISIBLE_DEVICES="0,1" python time_series_forecasting/training_vH_data_in_file.py --data_csv_path "data/X_train_qn_25K_001.txt" --feature_target_names_path "data/y_train_qn_25k_001.txt" --output_json_path "models/trained_config.json" --log_dir "models/ts_views_logs" --model_dir "models/ts_views_models" > training_log_vH12_sr001_SW_T_Ep75_BS128_lr2.txt

#python time_series_forecasting/training_vH.py --data_csv_path "data/X_train_qn_dummysmall.txt" --feature_target_names_path "data/y_train_qn_dummysmall.txt" --output_json_path "models/trained_config.json" --log_dir "models/ts_views_logs" --model_dir "models/ts_views_models" > training_log_vH5_dummysmall_v3.txt

#python time_series_forecasting/calculate_rmse_prep_plot_data.py

echo -en "\007"
echo "Completed   Completed   Completed"
date

#python3 test.py
