"""
    Define:
    + `||`: concatenate
    + `embed`: word embedding, e.g. through bert
    + `onehot`: one-hot encoding vector
    + other: directly as a number / vector

    Then:
    + user: onehot(gender) || embed(occupation_titles || interests || recreation_names)
    + course: log(price)+1 || onehot(subgroup) || embed(
        course_name || sub_groups || topics || will_learn || required_tools || recommended_background 
        || target_group || concat(chapter_item_name)
      )

    Note:
    + model(**inputs) will only work if inputs ONLY contain ['input_ids', 'token_type_ids', 'attention_mask']
      and ALL 3 is of shape (N, *). 
        + i.e. `inputs = pdss['user'][0]` won't work because its doesn't have the (**N**, ...) dimension,
          but `inputs = pdss['user'][0:1]` will work.
"""


from utils import *

import pandas as pd
from datasets import load_dataset

def testMain(args):
    """
        testing if we can actually make embedding from bert
        ref. https://is-rajapaksha.medium.com/bert-word-embeddings-deep-dive-32f6214f02bf
    """
    accelerator = Accelerator(fp16=args.fp16)

    accelerator.print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
    tokenizer = loadTokenizer(args)
    model = loadModel(args)
    model = accelerator.prepare(model)

    model.eval()
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    print(inputs)
    inputs = inputs.to(accelerator.device)
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state  # (1, 8, 768), 8 is probably token count
    print(last_hidden_states[:, 0, :])              # hopefully the first token's embedding
    print(last_hidden_states[:, 0, :].shape)        # (1, 768), so it works!

def testDataAndModel(args):
    """
        test if we nailed the dataset preprocess part
    """
    accelerator = Accelerator(fp16=args.fp16)
    accelerator.print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
    model = loadModel(args)

    tokenizer = loadTokenizer(args)
    dss = loadDataset(args)
    pdss = preprocessDataset(args, dss, tokenizer)
    """
        For reference:
        >>> pdss
        DatasetDict({
            user: Dataset({
                features: ['user_id', 'gender', 'occupation_titles', 'interests', 'recreation_names', 'input_ids', 'token_type_ids', 'attention_mask', 'gender_one_hot'],
                num_rows: 130566
            })
            item: Dataset({
                features: ['course_id', 'course_name', 'course_price', 'teacher_id', 'teacher_intro', 'groups', 'sub_groups', 'topics', 'course_published_at_local', 'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group', 'input_ids', 'token_type_ids', 'attention_mask'],
                num_rows: 728
            })
            subgroup: Dataset({
                features: ['subgroup_id', 'subgroup_name'],
                num_rows: 91
            })
            chapter: Dataset({
                features: ['course_id', 'chapter_no', 'chapter_id', 'chapter_name', 'chapter_item_id', 'chapter_item_no', 'chapter_item_name', 'chapter_item_type', 'video_length_in_seconds'],
                num_rows: 21290
            })
        })
    """

    # note that pdss is a dataset, not dataloader, thus cannot be prepared
    model = accelerator.prepare(model)

    model.eval()
    inputs = pdss['user'][0:1]  # contains one sample, DON'T use `pdss['user'][0]`
    outputs = model(**toBertInput(inputs, to_device=accelerator.device))

    last_hidden_states = outputs.last_hidden_state  # (1, token_cnt, 768)
    print(last_hidden_states[:, 0, :])
    print(last_hidden_states[:, 0, :].shape)        # (1, 768)

if __name__ == '__main__':
    args = parse_args()
    testDataAndModel(args)