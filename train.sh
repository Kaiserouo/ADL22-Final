

python two-towels.py \
  --task "train" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --load_tt_model_dir "./default_model_ep10" \
  --save_tt_model_dir "./default_model_ep30" \
  --num_epochs 20 