python two-towels.py \
  --two_towel_task "train" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --save_tt_model_dir "./models/def_w02_ep10" \
  --num_epochs 10 \
  --tt_ns_w 0.2

python two-towels.py \
  --two_towel_task "train" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --load_tt_model_dir "./models/def_w02_ep10" \
  --save_tt_model_dir "./models/def_w02_ep30" \
  --num_epochs 20 \
  --tt_ns_w 0.2

python two-towels.py \
  --two_towel_task "train" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --load_tt_model_dir "./models/def_w02_ep30" \
  --save_tt_model_dir "./models/def_w02_ep50" \
  --num_epochs 20 \
  --tt_ns_w 0.2