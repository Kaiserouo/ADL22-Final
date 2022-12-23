python -i two-towels.py \
  --we_model_name_or_path "hfl/chinese-roberta-wwm-ext" \
  --user_file "../hahow/data/users.csv" \
  --item_file "../hahow/data/courses.csv" \
  --chapter_file "../hahow/data/course_chapter_items.csv" \
  --subgroup_file "../hahow/data/subgroups.csv" \
  --load_feature_ds "./fdss.hf" \
  --load_useritem_example_ds "./eds.hf"