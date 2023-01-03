
# model directory path used for prediction, will also put prediction result in there
MODEL_PATH="./models/def_w02_ep50"
# hahow data directory path
DATA_PATH="../hahow/data"
# saved preprocessed dataset from feature.sh
EDS="./eds.hf"
FDSS="./fdss.hf"

# generates seen / unseen course prediction
# will store in ${MODEL_PATH}/rec_seen_course.csv and ${MODEL_PATH}/rec_unseen_course.csv
python two-towels.py \
  --two_towel_task "predict" \
  --load_feature_ds "${FDSS}" \
  --load_useritem_example_ds "${EDS}" \
  --load_tt_model_dir "${MODEL_PATH}" \
  --test_item_seen_file "${DATA_PATH}/test_seen.csv" \
  --test_item_unseen_file "${DATA_PATH}/test_unseen.csv" \
  --save_rec_path "${MODEL_PATH}" \
  --tt_ns_w 0.2

# generate subgroups by course prediction
# will store in ${MODEL_PATH}/rec_seen_subgroup.csv and ${MODEL_PATH}/rec_unseen_subgroup.csv
python ./generate_subgroup/seen_subgroup_baseline.py \
    --data_root "${DATA_PATH}" \
    --prev_course "${DATA_PATH}/train.csv" \
    --prev_course_2 "${MODEL_PATH}/rec_seen_course.csv" \
    --pred_file "${MODEL_PATH}/rec_seen_subgroup.csv"

python ./generate_subgroup/unseen_subgroup_baseline.py \
    --data_root "${DATA_PATH}" \
    --prev_course "${MODEL_PATH}/rec_unseen_course.csv" \
    --pred_file "${MODEL_PATH}/rec_unseen_subgroup.csv"