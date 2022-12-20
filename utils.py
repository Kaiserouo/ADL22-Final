import json
import csv
import argparse
from pathlib import Path
import os

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from transformers import (
    T5ForConditionalGeneration, set_seed,
    AutoConfig,
    BertTokenizer, BertModel
)
import math
from accelerate import Accelerator
from tqdm import tqdm
import datasets
from datasets import DatasetDict
import transformers

from collections import defaultdict

import numpy as np

class ArgumentManager:  
    @staticmethod
    def modelArguments(parser):
        group = parser.add_argument_group('Model importing')
        group.add_argument(
            "--model_name_or_path", type=str,
            help="Model name / path.",
        )
        group.add_argument(
            "--config_name", type=str, default=None,
            help="Config name / path. If is None then set to model_name_or_path",
        )
        group.add_argument(
            "--tokenizer_name", type=str, default=None,
            help="Tokenizer name / path. If is None then set to model_name_or_path",
        )
        group.add_argument(
            "--fp16", action="store_true",
            help="If you want to use fp16 or not.",
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

    @staticmethod
    def preprocessArguments(parser):
        group = parser.add_argument_group('Preprocessing')
        group.add_argument(
            "--max_length", type=int, default=256,
            help="Max length for tokenizer for input (text)",
        )

    @staticmethod
    def getAllFnLs():
        return [
            ArgumentManager.modelArguments,
            ArgumentManager.fileArguments,
            ArgumentManager.preprocessArguments,
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

def loadModel(args):
    # config
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError("Did not specify config")

    if args.model_name_or_path:
        model = BertModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        raise ValueError("Did not specify model")
    
    return model

def loadTokenizer(args):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        raise ValueError("Did not specify tokenizer")
    return tokenizer

def loadDataset(args):
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
        preprocess user examples to the format we want
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
        preprocess course
        subgroup_map: string -> index, 0-indexing!!!!!
        chapter map: item_id -> all chapter name as a string
        course_feature = log(price)+1 || onehot(subgroup) || embed(
            course_name || sub_groups || topics || will_learn || required_tools || recommended_background || target_group
        ) || concat(sorted(chapter_item_name, key=chapter_id))
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
            ) if subgroup != '' else torch.zeros(len(subgroup_map)).type(torch.LongTensor))
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
    model_inputs['course_price'] = torch.log(torch.Tensor(examples['course_price'])) + 1

    print([a.shape for a in subgroup_one_hot])
    model_inputs['subgroup_one_hot'] = subgroup_one_hot

    return model_inputs

def preprocessDataset(args, dss, tokenizer):
    # dss: all datasets

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
    
    dss['user'] = dss['user'].map(
        lambda x: preprocessUserExamples(x, tokenizer, args.max_length),
        batched=True
    )
    dss['item'] = dss['item'].map(
        lambda x: preprocessItemExamples(x, tokenizer, args.max_length, subgroup_map, chapter_map),
        batched=True
    )

    dss['user'].set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    dss['item'].set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])

    return dss

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