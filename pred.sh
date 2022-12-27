MODEL_PATH="./models/def_w02_ep100"  # will put the result in this tt_model_dir as well
# MODEL_PATH="${1}"  # will put the result in this tt_model_dir as well
DATA_PATH="../hahow/data"

python two-towels.py \
  --two_towel_task "predict" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf" \
  --load_tt_model_dir "${MODEL_PATH}" \
  --test_item_seen_file "../hahow/data/test_seen.csv" \
  --test_item_unseen_file "../hahow/data/test_unseen.csv" \
  --save_rec_path "${MODEL_PATH}" 

python ./generate_subgroup/seen_subgroup_baseline.py \
    --data_root "${DATA_PATH}" \
    --prev_course "${DATA_PATH}/train.csv" \
    --prev_course_2 "${MODEL_PATH}/rec_seen_course.csv" \
    --pred_file "${MODEL_PATH}/rec_seen_subgroup.csv"

python ./generate_subgroup/unseen_subgroup_baseline.py \
    --data_root "${DATA_PATH}" \
    --prev_course "${MODEL_PATH}/rec_unseen_course.csv" \
    --pred_file "${MODEL_PATH}/rec_unseen_subgroup.csv"