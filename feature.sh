DATA_PATH="../hahow/data"

python feature_extract.py \
  --feature_extract_task "eds" \
  --we_model_name_or_path "hfl/chinese-roberta-wwm-ext" \
  --user_file "${DATA_PATH}/users.csv" \
  --item_file "${DATA_PATH}/courses.csv" \
  --chapter_file "${DATA_PATH}/course_chapter_items.csv" \
  --subgroup_file "${DATA_PATH}/subgroups.csv" \
  --train_item_file "${DATA_PATH}/train.csv" \
  --save_eds "./eds.hf" \
  --save_fdss "./fdss.hf"

python feature_extract.py \
  --feature_extract_task "fdss" \
  --we_model_name_or_path "hfl/chinese-roberta-wwm-ext" \
  --user_file "${DATA_PATH}/users.csv" \
  --item_file "${DATA_PATH}/courses.csv" \
  --chapter_file "${DATA_PATH}/course_chapter_items.csv" \
  --subgroup_file "${DATA_PATH}/subgroups.csv" \
  --train_item_file "${DATA_PATH}/train.csv" \
  --save_eds "./eds.hf" \
  --save_fdss "./fdss.hf"