MODEL_PATH="./mdl_ep20"
DATA_PATH="../hahow/data"

python two-towels.py \
  --task "predict" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --load_tt_model_dir "${MODEL_PATH}" \
  --test_item_seen_file "../hahow/data/test_seen.csv" \
  --test_item_unseen_file "../hahow/data/test_unseen.csv" \
  --save_rec_path "${MODEL_PATH}" \
  --tt_dim_hidden 1024 \
  --tt_n_hidden 5 \
  --tt_p_dropout 0.15 \
  --tt_dim_encoding 728

python ./generate_subgroup/seen_subgroup_baseline.py \
    --data_root "${DATA_PATH}" \
    --prev_course "${DATA_PATH}/train.csv" \
    --prev_course_2 "${MODEL_PATH}/rec_seen_course.csv" \
    --pred_file "${MODEL_PATH}/rec_seen_subgroup.csv"

python ./generate_subgroup/unseen_subgroup_baseline.py \
    --data_root "${DATA_PATH}" \
    --prev_course "${MODEL_PATH}/rec_unseen_course.csv" \
    --pred_file "${MODEL_PATH}/rec_unseen_subgroup.csv"