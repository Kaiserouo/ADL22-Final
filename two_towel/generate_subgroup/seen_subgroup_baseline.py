import os
import pandas as pd
import numpy as np
from utils import mapk

import argparse

def process_data(data):
    for i in range(len(data)):
        if type(data[i]) == str:
            data[i] = data[i].split(' ')
        else:
            data[i] = [data[i]]
    return data

def create_dict(prev_courses, key='user_id', value='course_id', sep=' '):
    courses_dict = {}
    for i in range(len(prev_courses)):
        courses = str(prev_courses[value][i])
        if sep == None:
            courses_dict[prev_courses[key][i]] = courses
        else:
            if courses.find(sep) == -1:
                courses_dict[prev_courses[key][i]] = [courses]
            else:
                courses_dict[prev_courses[key][i]] = courses.split(sep)
    return courses_dict

def get_preds(courses_list, subgroups_dict, subgroups_id_dict):
    preds_list = []
    for courses in courses_list:
        preds_single = {}
        for course in courses:
            subgroups = subgroups_dict[course]
            if subgroups[0] == 'nan':
                continue
            for subgroup in subgroups:
                subgroup_id = subgroups_id_dict[subgroup]
                if preds_single.get(subgroup_id) == None:
                    preds_single[subgroup_id] = 1
                else:
                    preds_single[subgroup_id] += 1
        preds_single = sorted(preds_single.items(), key=lambda x: x[1], reverse=True)
        preds_list_single = [pred[0] for pred in preds_single]
        preds_list.append(preds_list_single)
    return preds_list

def main(args):
    # Read data
    prev_courses = pd.read_csv(args.prev_course)
    prev_courses_2 = pd.read_csv(args.prev_course_2)
    all_courses = pd.read_csv(os.path.join(args.data_root, 'courses.csv'), usecols=['course_id', 'sub_groups'])
    subgroups = pd.read_csv(os.path.join(args.data_root, 'subgroups.csv'))
    val = pd.read_csv(os.path.join(args.data_root, 'val_seen_group.csv'))
    test = pd.read_csv(os.path.join(args.data_root, 'test_seen_group.csv'))
    ans_list = process_data(list(val['subgroup']))

    # Data processing
    courses_dict = create_dict(prev_courses)
    courses_dict_2 = create_dict(prev_courses_2)
    subgroups_dict = create_dict(all_courses, key='course_id', value='sub_groups', sep=',')
    subgroups_id_dict = create_dict(subgroups, key='subgroup_name', value='subgroup_id', sep=None)

    ############################## Validation set ##############################
    # Get all users id and the courses id that they have taken
    # user_list_val = list(val['user_id'])
    # courses_list_val = []
    # for i in range(len(user_list_val)):
    #     courses_list_val.append(courses_dict[user_list_val[i]])
    #     if courses_dict_2.get(user_list_val[i]) != None:
    #         courses_list_val[i] += courses_dict_2[user_list_val[i]]
    # preds_list_val = get_preds(courses_list_val, subgroups_dict, subgroups_id_dict)
    # print("mAP@50 for validation set", mapk(ans_list, preds_list_val, k=50))

    ############################## Test set ##############################
    user_list_test = list(test['user_id'])
    courses_list_test = []
    for i in range(len(user_list_test)):
        courses_list_test.append(courses_dict[user_list_test[i]])
        if courses_dict_2.get(user_list_test[i]) != None:
             courses_list_test[i] += courses_dict_2[user_list_test[i]]
    preds_list_test = get_preds(courses_list_test, subgroups_dict, subgroups_id_dict)
    preds_list_test = [' '.join([str(pred) for pred in preds]) for preds in preds_list_test]
    test['subgroup'] = preds_list_test
    test.to_csv(args.pred_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--prev_course', type=str, default='train.csv')
    parser.add_argument('--prev_course_2', type=str, default='val_seen.csv')
    parser.add_argument('--pred_file', type=str, default='preds.csv')
    args = parser.parse_args()

    main(args)