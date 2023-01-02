import numpy as np
import pandas as pd
import os
import argparse

from utils import mapk

def main(args):
    prev_courses = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    all_courses = pd.read_csv(os.path.join(args.data_root, 'courses.csv'), usecols=['course_id', 'course_price'])
    if args.task == 'seen':
        val = pd.read_csv(os.path.join(args.data_root, 'val_seen.csv'))
        test = pd.read_csv(os.path.join(args.data_root, 'test_seen.csv'))
    elif args.task == 'unseen':
        val = pd.read_csv(os.path.join(args.data_root, 'val_unseen.csv'))
        test = pd.read_csv(os.path.join(args.data_root, 'test_unseen.csv'))
    
    # Build course_id to index dictionary
    course_id_to_idx = {}
    course_idx_to_id = {}
    for i in range(len(all_courses)):
        course_id_to_idx[all_courses['course_id'][i]] = i
        course_idx_to_id[i] = all_courses['course_id'][i]

    # Build user_id to index dictionary
    user_id_to_idx = {}
    user_idx_to_id = {}
    for i in range(len(prev_courses)):
        user_id_to_idx[prev_courses['user_id'][i]] = i
        user_idx_to_id[i] = prev_courses['user_id'][i]

    # Get the frequency of each course
    # course_freq = [0] * len(all_courses)
    # for i in range(len(prev_courses)):
    #     course_list = prev_courses['course_id'][i].split(' ')
    #     for j, course in enumerate(course_list):
    #         course_freq[course_id_to_idx[course]] += 1
    #         pass
    
    # for i in range(len(val)):
    #     course_list = val['course_id'][i].split(' ')
    #     for j, course in enumerate(course_list):
    #         course_freq[course_id_to_idx[course]] += 1
    # course_freq = np.array(course_freq)

    # Recommend the top 50 courses with lowest price
    top_50_idx = np.argsort(all_courses['course_price'].values)

    # Recommend the top 50 courses with highest frequency
    # top_50_idx = np.argsort(course_freq)[::-1]

    # Recommend these courses to all users
    preds_val = []
    for i in range(len(val)):
        pred = []
        for j in range(len(top_50_idx)):
            # Filter out the courses that the user has already taken in train.csv
            if args.task == 'seen' and course_idx_to_id[top_50_idx[j]] in prev_courses['course_id'][user_id_to_idx[val['user_id'][i]]].split(' '):
                continue
            else:
                pred.append(course_idx_to_id[top_50_idx[j]])
            if len(pred) == 50:
                break
        # Sort the recommended courses by price
        # pred = sorted(pred, key=lambda x: all_courses['course_price'][course_id_to_idx[x]])
        preds_val.append(pred)

    # Get ground truth
    val_ground_truth = []
    val_user_id2idx = {}
    for i in range(len(val)):
        val_user_id2idx[val['user_id'][i]] = i
        val_ground_truth.append(val['course_id'][i].split(' '))

    # Calculate MAP@50
    val_mapk = mapk(val_ground_truth, preds_val, 50)
    print('MAP@50 on validation set: {}'.format(val_mapk))

    # Predict on test set and save the results
    pred_test = []
    for i in range(len(test)):
        pred = []
        for j in range(len(top_50_idx)):
            # Filter out the courses that the user has already taken in train.csv and val.csv
            if args.task == 'seen' and course_idx_to_id[top_50_idx[j]] in prev_courses['course_id'][user_id_to_idx[test['user_id'][i]]].split(' '):
                continue
            elif val_user_id2idx.get(test['user_id'][i], -1) != -1 and course_idx_to_id[top_50_idx[j]] in val_ground_truth[val_user_id2idx[test['user_id'][i]]]:
                continue
            else:
                pred.append(course_idx_to_id[top_50_idx[j]])
            if len(pred) == 50:
                break
        # Sort the recommended courses by price
        # pred = sorted(pred, key=lambda x: all_courses['course_price'][course_id_to_idx[x]])
        pred_test.append(' '.join(pred))
    test['course_id'] = pred_test
    fname = f'rule-based-{args.task}.csv'
    test.to_csv(os.path.join(args.output_path, fname), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--task', type=str, default='seen')
    parser.add_argument('--output_path', type=str, default='me')
    args = parser.parse_args()

    main(args)