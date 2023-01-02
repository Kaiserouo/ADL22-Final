import pandas as pd
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import mapk

class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        # trainable lookup matrix for shallow embedding vectors
        self.user_embed = nn.Embedding(n_users, 32)
        self.movie_embed = nn.Embedding(n_movies, 32)

        # MLP layers with dropout and batchnorm
        self.MLP = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
        )

        # user, course embedding concat
        self.out = nn.Linear(64, 1)

    
    def forward(self, users, movies, ratings=None):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        output = torch.cat([user_embeds, movie_embeds], dim=1)
        
        output = self.MLP(output)
        output = self.out(output)
        
        return output

class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    # len(movie_dataset)
    def __len__(self):
        return len(self.users)
    # movie_dataset[1] 
    def __getitem__(self, item):

        users = self.users[item] 
        movies = self.movies[item]
        ratings = self.ratings[item]
        
        return {
            "users": torch.tensor(users, dtype=torch.long),
            "courses": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float32),
        }

def main(args):
    prev_courses = pd.read_csv(os.path.join(args.data_root, 'train.csv'))
    all_courses = pd.read_csv(os.path.join(args.data_root, 'courses.csv'), usecols=['course_id', 'course_price'])
    val = pd.read_csv(os.path.join(args.data_root, 'val_seen.csv'))
    test = pd.read_csv(os.path.join(args.data_root, 'test_seen.csv'))
    
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

    if args.pred_mode:
        model = RecSysModel(len(user_id_to_idx), len(course_id_to_idx))
        model.load_state_dict(torch.load(args.model_path)['model'])
        model.eval()
        user_test = []
        course_test = []
        for i in range(len(test)):
            for j in range(len(all_courses)):
                user_test.append(user_id_to_idx[test['user_id'][i]])
                course_test.append(course_id_to_idx[all_courses['course_id'][j]])
        test_dataset = MovieDataset(user_test, course_test, [0] * len(user_test))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
        with torch.no_grad():
            user_course_matrix_pred = np.zeros((len(prev_courses), len(all_courses)))
            for batch in tqdm(test_loader):
                users = batch['users']
                courses = batch['courses']
                preds = model(users, courses)
                preds = preds.cpu().numpy()
                for i in range(len(users)):
                    user_course_matrix_pred[users[i]][courses[i]] = preds[i]
            predictions = user_course_matrix_pred

            # Get top 50 courses and save to csv
            for i in range(len(test)):
                user_id_test = test['user_id'][i]
                user_idx_test = user_id_to_idx[user_id_test]

                # Filter out courses that user has already seen
                course_id_seen = prev_courses['course_id'][user_idx_test].split(' ')
                course_idx_seen = [course_id_to_idx[course_id] for course_id in course_id_seen]
                predictions[user_idx_test][course_idx_seen] = -np.inf

                course_idx_test = np.argsort(predictions[user_idx_test])[::-1][:50]
                course_id_test = [course_idx_to_id[idx] for idx in course_idx_test]
                course_id_test = ' '.join(course_id_test)
                test.loc[i, 'course_id'] = course_id_test
            test.to_csv(args.output_path, index=False)
        return

    # Build course frequency list
    course_freq = [0] * len(all_courses)
    for i in range(len(prev_courses)):
        course_list = prev_courses['course_id'][i].split(' ')
        for course in course_list:
            course_freq[course_id_to_idx[course]] += 1
    course_freq = np.array(course_freq)
    high_freq_idx = np.where(course_freq > 1000)[0]
    
    # Get all the courses with price == 0
    free_courses = all_courses[all_courses['course_price'] == 0]['course_id'].values
    free_courses_idx = [course_id_to_idx[course] for course in free_courses]

    # Build dataset
    users = []
    courses = []
    ratings = []
    for i in range(len(prev_courses)):
        course_list = prev_courses['course_id'][i].split(' ')
        for course in course_list:
            if course_freq[course_id_to_idx[course]] > 1000:
                continue
            # Append one-hot encoded user vector
            users.append(user_id_to_idx[prev_courses['user_id'][i]])
            # Append one-hot encoded course vector
            courses.append(course_id_to_idx[course])
            ratings.append(1 * np.log10(1 + course_freq[course_id_to_idx[course]]))

        # Add high frequency courses to training set
        for course in high_freq_idx:
            if course in free_courses_idx:
                users.append(user_id_to_idx[prev_courses['user_id'][i]])
                courses.append(course)
                ratings.append(1 * np.log10(1 + course_freq[course]))

    user_val = []
    course_val = []
    for i in range(len(val)):
        for j in range(len(all_courses)):
            user_val.append(user_id_to_idx[val['user_id'][i]])
            course_val.append(course_id_to_idx[all_courses['course_id'][j]]) 
    
    dataset = MovieDataset(users, courses, ratings)
    dataset_val = MovieDataset(user_val, course_val, [0] * len(user_val))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size_val, shuffle=False, num_workers=4)

    # Build model
    model = RecSysModel(len(prev_courses), len(all_courses))
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.MSELoss()

    # Train model
    model.train()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        losses = []
        for batch in tqdm(train_loader):
            users = batch["users"].to(args.device)
            courses = batch["courses"].to(args.device)
            ratings = batch["ratings"].to(args.device)
            optimizer.zero_grad()
            output = model(users, courses)
            loss = criterion(output, ratings.view(-1, 1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        scheduler.step()
        print(f"Epoch {epoch}: {np.mean(losses)}")

        # Save model
        model_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(model_dict, os.path.join(args.save_dir, f"model_{epoch}.pt"))

    print("Validation")
    with torch.no_grad():
        model.eval()
        user_course_matrix_pred = np.zeros((len(prev_courses), len(all_courses)))
        for batch in tqdm(val_loader):
            users = batch["users"].to(args.device)
            courses = batch["courses"].to(args.device)
            ratings = batch["ratings"].to(args.device)
            output = model(users, courses)
            
            # Store the prediction in a matrix
            for i in range(len(users)):
                user_course_matrix_pred[users[i]][courses[i]] = output[i].item()

        # Calculate MAP@50 for validation set
        val_pred = []
        for i in range(len(val)):
            user_id = val['user_id'][i]
            user_idx = user_id_to_idx[user_id]
            pred = user_course_matrix_pred[user_idx]

            # Filter out the courses which the user has taken
            course_list = prev_courses['course_id'][user_idx].split(' ')
            for course in course_list:
                pred[course_id_to_idx[course]] = 0
            # Sort the prediction and get the top 50 courses
            pred = pred.argsort()[-50:][::-1]
            pred = [course_idx_to_id[idx] for idx in pred]
            val_pred.append(pred)

        # Calculate MAP@50 for validation set
        val_true = []
        for i in range(len(val)):
            course_list = val['course_id'][i].split(' ')
            val_true.append(course_list)
        val_map = mapk(val_true, val_pred, k=50)
        print('Validation MAP@50: {}'.format(val_map))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='model')
    parser.add_argument('--model_path', type=str, default='model/model_best_type2.pt')
    parser.add_argument('--output_path', type=str, default='me/pred.csv')
    parser.add_argument('--pred_mode', action='store_true')

    # Model parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--batch_size_val', type=int, default=4096)
    parser.add_argument('--random_seed', type=int, default=0)

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    if args.pred_mode and args.model_path is None:
        raise ValueError('Please specify the model path for prediction mode')

    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check cuda
    if torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        args.device = torch.device('cpu')

    print('using device: ', args.device)
    main(args)