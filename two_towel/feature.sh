# data directory for hahow, should contain files specified below
DATA_PATH="../hahow/data"

# save preprocessed datasets path, should plug in train.sh and pred.sh for the script to work properly
# EDS: example dataset, a dataset that contains (user, item, label) pairs for training
#      (label is 1 if user bought the item, 0 if not)
#      ref. utils.py's makeUserItemExampleDataset()
SAVE_EDS="./eds.hf"

# FDSS: feature datasets, contains dataset of user and item embeddings
#       in report, it's called user / item feature vector
#       ref. utils.py's makeUserItemFeatureDataset()
SAVE_FDSS="./fdss.hf"

python feature_extract.py \
  --feature_extract_task "eds" \
  --we_model_name_or_path "hfl/chinese-roberta-wwm-ext" \
  --user_file "${DATA_PATH}/users.csv" \
  --item_file "${DATA_PATH}/courses.csv" \
  --chapter_file "${DATA_PATH}/course_chapter_items.csv" \
  --subgroup_file "${DATA_PATH}/subgroups.csv" \
  --train_item_file "${DATA_PATH}/train.csv" \
  --save_eds "${SAVE_EDS}" \
  --save_fdss "${SAVE_FDSS}"

python feature_extract.py \
  --feature_extract_task "fdss" \
  --we_model_name_or_path "hfl/chinese-roberta-wwm-ext" \
  --user_file "${DATA_PATH}/users.csv" \
  --item_file "${DATA_PATH}/courses.csv" \
  --chapter_file "${DATA_PATH}/course_chapter_items.csv" \
  --subgroup_file "${DATA_PATH}/subgroups.csv" \
  --train_item_file "${DATA_PATH}/train.csv" \
  --save_eds "${SAVE_EDS}" \
  --save_fdss "${SAVE_FDSS}"