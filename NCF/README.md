# ADL-Final

## Training
```shell
python matrix_factor_model_dl.py --data_root ${hahow_data_dir} --save_dir ${path_to_save_trained_models} --lr ...
```

## Predicting
```shell
python matrix_factor_model_dl.py --data_root ${hahow_data_dir} --model_path ${pretrained_model_path} --output_path ${path_to_save_kaggle_prediction} --pred_mode
```