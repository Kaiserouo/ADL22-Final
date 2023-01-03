
# saved preprocessed dataset from feature.sh
EDS="./eds.hf"
FDSS="./fdss.hf"

# saved model path, this will be a directory that contains the model 
SAVE_MODEL_PATH="./models/def_w02_ep50"

# main script, specify your model parameter as you like
python two-towels.py \
  --two_towel_task "train" \
  --load_feature_ds "${FDSS}" \
  --load_useritem_example_ds "${EDS}" \
  --save_tt_model_dir "${SAVE_MODEL_PATH}" \
  --num_epochs 50 \
  --tt_ns_w 0.2