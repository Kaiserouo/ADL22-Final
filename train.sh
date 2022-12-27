python two-towels.py \
  --two_towel_task "train" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --save_tt_model_dir "./models/1024_w02_ep50" \
  --num_epochs 50 \
  --tt_dim_hidden 1024 \
  --tt_n_hidden 5 \
  --tt_p_dropout 0.15 \
  --tt_dim_encoding 728 \
  --tt_ns_w 0.2

python two-towels.py \
  --two_towel_task "train" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --load_tt_model_dir "./models/1024_w02_ep50" \
  --save_tt_model_dir "./models/1024_w02_ep100" \
  --num_epochs 50 \
  --tt_dim_hidden 1024 \
  --tt_n_hidden 5 \
  --tt_p_dropout 0.15 \
  --tt_dim_encoding 728 \
  --tt_ns_w 0.2