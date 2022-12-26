python feature_extract.py \
  --feature_extract_task "eds" \
  --we_model_name_or_path "hfl/chinese-roberta-wwm-ext" \
  --user_file "../hahow/data/users.csv" \
  --item_file "../hahow/data/courses.csv" \
  --chapter_file "../hahow/data/course_chapter_items.csv" \
  --subgroup_file "../hahow/data/subgroups.csv" \
  --train_item_file "../hahow/data/train.csv" \
  --save_eds "./eds.hf" \
  --save_fdss "./fdss.hf"

python feature_extract.py \
  --feature_extract_task "fdss" \
  --we_model_name_or_path "hfl/chinese-roberta-wwm-ext" \
  --user_file "../hahow/data/users.csv" \
  --item_file "../hahow/data/courses.csv" \
  --chapter_file "../hahow/data/course_chapter_items.csv" \
  --subgroup_file "../hahow/data/subgroups.csv" \
  --train_item_file "../hahow/data/train.csv" \
  --save_eds "./eds.hf" \
  --save_fdss "./fdss.hf"