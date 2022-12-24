import json
import csv
import argparse
from pathlib import Path
import os
import math
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random

import datasets
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import torch
from transformers import (
    T5ForConditionalGeneration, set_seed,
    AutoConfig,
    BertTokenizer, BertModel, 
    AutoTokenizer
)
import transformers
from accelerate import Accelerator

class ArgumentManager:  
    @staticmethod
    def globalModelArguments(parser):
        group = parser.add_argument_group('Global model parameter that will suit on all model.')
        group.add_argument(
            "--fp16", action="store_true",
            help="If you want to use fp16 or not.",
        )

    @staticmethod
    def wwModelArguments(parser):
        group = parser.add_argument_group('Word embedding model')
        group.add_argument(
            "--we_model_name_or_path", type=str,
            help="Word embedding model name / path.",
        )
        group.add_argument(
            "--we_config_name", type=str, default=None,
            help="Word embedding config name / path. If is None then set to model_name_or_path",
        )
        group.add_argument(
            "--we_tokenizer_name", type=str, default=None,
            help="Word embedding tokenizer name / path. If is None then set to model_name_or_path",
        )
    
    @staticmethod
    def ttModelArguments(parser):
        group = parser.add_argument_group('Two towel model')
        group.add_argument(
            "--tt_dim_hidden", type=int, default=500,
            help="Hidden dim inside two towel model's EncodingModel",
        )
        group.add_argument(
            "--tt_n_hidden", type=int, default=2,
            help="Number of hidden layers inside EncodingModel",
        )
        group.add_argument(
            "--tt_p_dropout", type=float, default=0.1,
            help="Dropout probability of EncodingModel",
        )
        group.add_argument(
            "--tt_dim_encoding", type=int, default=500,
            help="Encoding dim of two towel model. Will be the final user/item embedding dimension.",
        )

    @staticmethod
    def fileArguments(parser):
        group = parser.add_argument_group('Files')
        group.add_argument(
            "--user_file", type=str, default=None,
            help="User feature file. i.e. users.csv",
        )
        group.add_argument(
            "--item_file", type=str, default=None,
            help="Item / course feature file. i.e. courses.csv",
        )
        group.add_argument(
            "--chapter_file", type=str, default=None,
            help="Chapter for course. i.e. course_chapter_items.csv",
        )
        group.add_argument(
            "--subgroup_file", type=str, default=None,
            help="Mapping between subgroup and index. i.e. subgroups.csv",
        )
        group.add_argument(
            "--test_item_seen_file", type=str, default=None,
            help="i.e. test_seen.csv",
        )
        group.add_argument(
            "--test_item_unseen_file", type=str, default=None,
            help="i.e. test_unseen.csv",
        )

    @staticmethod
    def saveLoadArguments(parser):
        group = parser.add_argument_group('Saving & Loading Files')
        group.add_argument(
            "--load_feature_ds", type=str, default=None,
            help="Feature dataset loading path",
        )
        group.add_argument(
            "--load_useritem_example_ds", type=str, default=None,
            help="User item example dataset loading path",
        )
        group.add_argument(
            "--load_tt_model_dir", type=str, default=None,
            help="Load directory for two-towel model. Used when predicting or continuing training",
        )
        group.add_argument(
            "--save_tt_model_dir", type=str, default=None,
            help="Save directory for two-towel model, and its parameters",
        )
        group.add_argument(
            "--save_rec_path", type=str, default=None,
            help="Save path for recommendation result. Should be a directory",
        )

    @staticmethod
    def preprocessArguments(parser):
        group = parser.add_argument_group('Preprocessing')
        group.add_argument(
            "--max_length", type=int, default=256,
            help="Max length for tokenizer for input (text)",
        )

    @staticmethod
    def trainingArguments(parser):
        group = parser.add_argument_group('Training')
        group.add_argument(
            "--train_batch_size", type=int, default=512,
            help="Training batch size",
        )
        group.add_argument(
            "--eval_batch_size", type=int, default=8,
            help="Evaluation batch size",
        )
        group.add_argument(
            "--tt_ns_w", type=float, default=0.1,
            help="Weight given to negative sample in weighted matrix factorization loss function",
        )
        group.add_argument(
            "--num_epochs", type=int, default=10,
            help="Number of training epochs",
        )
        group.add_argument(
            "--learning_rate", type=float, default=1e-3,
            help="Learning rate",
        )

    @staticmethod
    def taskArguments(parser):
        group = parser.add_argument_group('Tasks, must-have')
        group.add_argument(
            "--task", type=str, default="train",
            help="Task specifier, default to 'train'.",
            choices=["train", "predict"]
        )

    @staticmethod
    def getAllFnLs():
        return [
            ArgumentManager.globalModelArguments,
            ArgumentManager.wwModelArguments,
            ArgumentManager.ttModelArguments,
            ArgumentManager.fileArguments,
            ArgumentManager.saveLoadArguments,
            ArgumentManager.preprocessArguments,
            ArgumentManager.trainingArguments,
            ArgumentManager.taskArguments,
        ]

def parse_args(fn_ls=None):
    "parse all arguments with all functions in fn_ls, which will all be called with args = [parser]."
    parser = argparse.ArgumentParser(description="Turn user and item CSV into features")
    if fn_ls is None:
        fn_ls = ArgumentManager.getAllFnLs()
    for fn in fn_ls:
        fn(parser)
    args = parser.parse_args()

    return args

def setAcceleratorLoggingVerbosity(accelerator):
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

def loadModelStr(model_name_or_path, config_name):
    "Just a version that does not just take the argument"
    # config
    if config_name:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        raise ValueError("Did not specify config")

    if model_name_or_path:
        model = BertModel.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )
    else:
        raise ValueError("Did not specify model")
    
    return model

def loadModelWE(args):
    "load word embedding model (usually bert)"
    return loadModelStr(args.we_model_name_or_path, args.we_config_name)

def loadTokenizerWE(args):
    if args.we_tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.we_tokenizer_name, use_fast=True)
    elif args.we_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.we_model_name_or_path, use_fast=True)
    else:
        raise ValueError("Did not specify tokenizer")
    return tokenizer

def loadDataset(args):
    """
        Load raw datasets, directly from csv files. IDs are strings
        For reference:
        >>> loadDataset(args)
        DatasetDict({
            user: Dataset({
                features: ['user_id', 'gender', 'occupation_titles', 'interests', 'recreation_names'],
                num_rows: 130566
            })
            item: Dataset({
                features: ['course_id', 'course_name', 'course_price', 'teacher_id', 'teacher_intro', 'groups', 'sub_groups', 'topics', 'course_published_at_local', 'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group'],
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
    data_files = {
        'user': args.user_file,
        'item': args.item_file,
        'subgroup': args.subgroup_file,
        'chapter': args.chapter_file
    }
    return datasets.DatasetDict(
        {
            name: load_dataset("csv", data_files=fname)['train']
            for name, fname in data_files.items()
        }
    )
    
def substituteNone(s, *lss):
    # for ls in lss, substitute None with s
    return [[s if i is None else i for i in ls] for ls in lss]    

def preprocessUserExamples(examples, tokenizer, max_length, sep="|"):
    """
        preprocess user examples to the format we want, used for ds['user'].map()
        just tokenize, don't go through embedding model and concat all tensors
        user_feature = onehot(gender) || embed(occupation_titles || interests || recreation_names)
    """
    # constants
    gender_map = {
        None: 0,
        'female': 1,
        'male': 2,
        'other': 3
    }

    # needed fields
    gender = examples['gender']
    occupation_titles = examples['occupation_titles']
    interests = examples['interests']
    recreation_names = examples['recreation_names']
    
    # preprocess
    occupation_titles, interests, recreation_names = substituteNone(
        "", 
        occupation_titles, interests, recreation_names
    )

    # one-hot
    gender = torch.LongTensor([gender_map[g] for g in gender])
    gender_one_hot = torch.nn.functional.one_hot(gender, num_classes=len(gender_map))

    # tokenize
    inputs = [sep.join(tp) for tp in zip(occupation_titles, interests, recreation_names)]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    model_inputs['gender_one_hot'] = gender_one_hot
    
    return model_inputs

def preprocessItemExamples(examples, tokenizer, max_length, subgroup_map, chapter_map, sep="|"):
    """
        preprocess course, used for ds['item'].map()
        subgroup_map: string -> index, 0-indexing!!!!!
        chapter map: item_id -> all chapter name as a string
        course_feature: log(price + 1) || onehot(subgroup) || embed(
            course_name || sub_groups || topics || will_learn || required_tools || recommended_background 
            || target_group || concat(chapter_item_name)
        )
    """
    course_name = examples['course_name']
    sub_groups = examples['sub_groups']
    topics = examples['topics']
    will_learn = examples['will_learn']
    required_tools = examples['required_tools']
    recommended_background = examples['recommended_background']
    target_group = examples['target_group']

    # preprocess
    course_name, sub_groups, topics, will_learn, required_tools, recommended_background, target_group = substituteNone(
        "",
        course_name, sub_groups, topics, will_learn, required_tools, recommended_background, target_group
    )

    # one-hot for subgroup
    # if a course have a subgroup, then its position will be one
    # so for courses with multiple subgroup, multiple position will be one
    subgroup_one_hot = torch.stack([
        torch.sum(
            (torch.nn.functional.one_hot(
                torch.LongTensor([subgroup_map[sg] for sg in subgroup.split(',')]),
                num_classes=len(subgroup_map)
            ) if subgroup != '' else torch.zeros((1, len(subgroup_map))).type(torch.LongTensor))
            , dim=0
        )
        for subgroup in sub_groups
    ])

    chapter_names = [chapter_map[course_id] for course_id in examples['course_id']]

    # tokenize
    inputs = [
        sep.join(tp)
        for tp in zip(
            course_name, sub_groups, topics, will_learn, required_tools, recommended_background, 
            target_group, chapter_names
        )
    ]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)

    # other
    model_inputs['course_price'] = torch.log(torch.Tensor(examples['course_price']) + 1)

    model_inputs['subgroup_one_hot'] = subgroup_one_hot

    return model_inputs

def preprocessDataset(args, dss, tokenizer=None):
    """
        preprocess datasets
        
        dss: contain at least [user, item, subgroup, chapter]. IDs are strings
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

    if tokenizer is None:
        tokenizer = loadTokenizerWE(args)

    # make subgroup map
    subgroup_map = dict()
    for subgroup in dss['subgroup']:
        # subtract 1 to start from 0
        subgroup_map[subgroup['subgroup_name']] = subgroup['subgroup_id'] - 1

    # make chapter map
    chapter_map = defaultdict(list)
    for chapter in dss['chapter']:
        chapter_map[chapter['course_id']].append(
            chapter['chapter_item_name'] if chapter['chapter_item_name'] is not None else ""
        )
    chapter_map = defaultdict(str, {course_id: ','.join(ls) for course_id, ls in chapter_map.items()})
    
    pdss = DatasetDict()
    pdss['user'] = dss['user'].map(
        lambda x: preprocessUserExamples(x, tokenizer, args.max_length),
        batched=True,
        desc="preprocessDataset: User"
    )
    pdss['item'] = dss['item'].map(
        lambda x: preprocessItemExamples(x, tokenizer, args.max_length, subgroup_map, chapter_map),
        batched=True,
        desc="preprocessDataset: Item"
    )

    pdss['user'].set_format(
        type="torch", 
        columns=["input_ids", "token_type_ids", "attention_mask", "gender_one_hot"],
        output_all_columns=True
    )
    pdss['item'].set_format(
        type="torch", 
        columns=["input_ids", "token_type_ids", "attention_mask", "subgroup_one_hot", "course_price"],
        output_all_columns=True
    )

    return pdss

def toBertInput(inputs, to_device=None):
    """
        model(**inputs) won't work if input contains other fields,
        model(**toBertInput(inputs)) solves this problem
    """
    if to_device is None:
        return {
            name: inputs[name]
            for name in ['input_ids', 'token_type_ids', 'attention_mask']
        }
    else:
        return {
            name: inputs[name].to(to_device)
            for name in ['input_ids', 'token_type_ids', 'attention_mask']
        }

def makeUserItemExamplesDataset(dss, df, neg_portion=0.5):
    """
        make some positive and negative examples for user-item pairs

        dss: dataset (containing at least user & item)
        df: dataframe for positive `user_id -> list of item_id`, e.g. train.csv
        neg_portion: how much portion does negative samples have, in [0, 1)
            e.g. if n_pos = 100 and neg_portion = 0.25, then we will make 33 negative examples (33/(100+33) = 0.25)
            this can be set to 0 to avoid making any negative examples

        return a dataset containing [user_id, item_id, label (0 or 1)], not sorted in any way
        IDs are strings
    """
    user_ids = dss['user']['user_id']
    item_ids = dss['item']['course_id']

    pos_set = set()
    
    # get all positive samples
    for i, row in df.iterrows():
        user_id = row['user_id']
        for item_id in row['course_id'].split(' '):
            pos_set.add((user_id, item_id))
    
    n_pos = len(pos_set)
    n_neg = int(n_pos * neg_portion / (1 - neg_portion))
    
    # sample negatives
    neg_set = set()
    while len(neg_set) < n_neg:
        user_id = random.choice(user_ids)
        item_id = random.choice(item_ids)
        p = (user_id, item_id)
        if p in pos_set or p in neg_set:
            continue
        neg_set.add(p)
    
    # make dataset
    ds_ls = [{'user_id': uid, 'item_id': iid, 'label': 1} for uid, iid in pos_set]
    ds_ls.extend([{'user_id': uid, 'item_id': iid, 'label': 0} for uid, iid in neg_set])
    
    return Dataset.from_list(ds_ls)

def loadUserItemExamplesDataset(args):
    return datasets.load_from_disk(args.load_useritem_example_ds)

def makeUserItemFeatureDataset(args, accelerator, pdss):
    """
        make user and item feature vectors, i.e. actually go through stuff because I
        don't want to train bert for now. Will add an `embed` column, which is the feature vector
        
        IDs are still strings

        pdss: from preprocessDataset()

        return DatasetDict({
            user: Dataset({
                features: ['user_id', 'embed'],
                num_rows: 130566
            })
            item: Dataset({
                features: ['item_id', 'embed'],
                num_rows: 728
            })
        })

        where user embedding dim is 772, item is 860.
    """
    def userMap(examples, model, accelerator):
        "user_feature = onehot(gender) || embed(occupation_titles || interests || recreation_names)"
        outputs = model(**toBertInput(examples, to_device=accelerator.device))
        last_hidden_states = outputs.last_hidden_state  # (N, token_cnt, 768)
        embed = torch.concat((examples['gender_one_hot'].to(accelerator.device), last_hidden_states[:, 0, :]), dim=-1)
        # print(embed.shape)  # should be (N, 768+4)

        return {'user_id': examples['user_id'], 'embed': embed}


    def itemMap(examples, model, accelerator):
        """
            course_feature = log(price + 1) || onehot(subgroup) || embed(
                course_name || sub_groups || topics || will_learn || required_tools || recommended_background 
                || target_group || concat(chapter_item_name)
            )
        """
        outputs = model(**toBertInput(examples, to_device=accelerator.device))
        last_hidden_states = outputs.last_hidden_state  # (1, token_cnt, 768)
        embed = torch.concat(
            (
                examples['course_price'].unsqueeze(1).to(accelerator.device), 
                examples['subgroup_one_hot'].to(accelerator.device), 
                last_hidden_states[:, 0, :]
            ), dim=-1
        )

        return {'item_id': examples['course_id'], 'embed': embed}

    model = loadModelWE(args)
    model = accelerator.prepare(model)
    model.eval()
    ds_dict = DatasetDict()

    # 3060 6G, 20min
    ds_dict['user'] = pdss['user'].map(
        lambda x: userMap(x, model, accelerator),
        batched=True,
        batch_size=24,
        remove_columns=pdss['user'].column_names,
        desc="makeUserItemFeatureDataset: User"
    )
    ds_dict['item'] = pdss['item'].map(
        lambda x: itemMap(x, model, accelerator),
        batched=True,
        batch_size=8,
        remove_columns=pdss['item'].column_names,
        desc="makeUserItemFeatureDataset: Item"
    )
    return ds_dict

def loadUserItemFeatureDataset(args):
    return datasets.load_from_disk(args.load_feature_ds)
    
def makeUserItemEmbedding(fdss):
    """
        make embedding from fdss['user'], fdss['item']
        fdss is from makeFeatureDataset(), or load yourself one
        
        user_map, item_map: id(str) -> index(int), 0-indexing

        so for a user_id (str), its embedding is user_embed(item_map[user_id])
    """
    fdss['user'].set_format(
        type="torch", columns=["embed"], output_all_columns=True
    )
    fdss['item'].set_format(
        type="torch", columns=["embed"], output_all_columns=True
    )

    user_embed = torch.nn.Embedding.from_pretrained(fdss['user']['embed'], freeze=True)
    uids = fdss['user']['user_id']
    user_map = {uid: i for i, uid in enumerate(uids)}

    item_embed = torch.nn.Embedding.from_pretrained(fdss['item']['embed'], freeze=True)
    iids = fdss['item']['item_id']
    item_map = {iid: i for i, iid in enumerate(iids)}

    return user_embed, user_map, item_embed, item_map

def makeUserItemDataloader(args, eds, user_embed, user_map, item_embed, item_map, is_train=True):
    """
        make dataloader from eds,
        the resulting dataset inside will have ['user_embed', 'item_embed', 'label'] inside

        >>> a = iter(dataloader)    # batch size 8
        >>> b = next(a)
        >>> b['item_embed'].shape
        torch.Size([8, 860])
        >>> b['user_embed'].shape
        torch.Size([8, 772])
        >>> b['label'].shape
        torch.Size([8])
    """
    def embedMap(examples):
        # turn string id into number
        user_ids = [user_map[uid] for uid in examples['user_id']]
        item_ids = [item_map[iid] for iid in examples['item_id']]

        # turn id into embedding
        ret = dict()
        ret['user_embed'] = user_embed(torch.LongTensor(user_ids))
        ret['item_embed'] = item_embed(torch.LongTensor(item_ids))
        ret['label'] = examples['label']
        return ret

    ds = eds.map(
        embedMap, batched=True, remove_columns=eds.column_names,
        desc="makeUserItemDataloader"
    )

    ds.set_format(type="torch", columns=ds.column_names, output_all_columns=True)

    # make that into dataloader
    dataloader = DataLoader(
        ds,
        shuffle=True if is_train else False,
        batch_size=args.train_batch_size if is_train else args.eval_batch_size,
    )

    return dataloader