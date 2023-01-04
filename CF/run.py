import implicit
import pandas as pd
import numpy as np
import logging
import os

from typing import List, Dict, Any, Tuple
from scipy.sparse import csr_matrix
from argparse import ArgumentParser

from metric import mapk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class Hahow_Dataset():
    def __init__(self, paths: Dict[str, str], course_file: str) -> None:
        df = pd.read_csv(course_file)
        self.prices = {course: price for (course, price) in zip(df["course_id"], df["course_price"])}
        
        self._prepare_df(paths)

    def _prepare_df(self, paths):
        # A user may buy more than one courses, expand to the rows each of has exactly one user and one course.
        # Also, add price column.
        df_train = pd.read_csv(paths["train"])
        df_train["course_id"] = df_train["course_id"].str.split()
        df_train = df_train.explode("course_id", ignore_index=True)
        df_train["price"] = df_train["course_id"].map(self.prices)

        self._df = {}
        self._df["train"] = df_train
        if "eval" in paths:
            self._df["eval"] = pd.read_csv(paths["eval"])
            self._df["eval"]["course_id"] = self._df["eval"]["course_id"].str.split()

        if "test" in paths:
            self._df["test"] = pd.read_csv(paths["test"])
        

    def __getitem__(self, key):
        # key: "train", "eval", "test"
        return self._df[key] if key in self._df else None

class Implicit_Model():
    MODELS = {
        "als": implicit.als.AlternatingLeastSquares,
        "lmf": implicit.lmf.LogisticMatrixFactorization,
        "bpr": implicit.bpr.BayesianPersonalizedRanking,
    }
    
    def __init__(self, args) -> None:
        self.model_name = args.model_name

        model_class = self.MODELS.get(self.model_name, self.MODELS["als"])
        params = {
            "factors": args.factors,
            "iterations": args.iterations,
        }
        if args.regularization:
            params["regularization"] = args.regularization

        if self.model_name == "als":
            params["calculate_training_loss"] = True
        elif self.model_name == "bpr":
            params["learning_rate"] = args.lr
        elif self.model_name == "lmf":
            #params["neg_prop"] = args.neg_prop
            params["learning_rate"] = args.lr


        self.model = model_class(**params)
    
    def __call__(self, df: pd.DataFrame):
        self._create_ui_matrix(df)

        # Train!
        self.model.fit(self.user_item_matrix)
    
    def _create_ui_matrix(self, df: pd.DataFrame):
        # Create user-item matrix     
        # The creation depends on algorithm  
        users = df["user_id"].astype("category")
        items = df["course_id"].astype("category")

        # Important !!
        # Different strategy to put weight in user-item matrix
        #values = [1 for _ in range(df.shape[0])]
        #values = (1/(df["price"]+1)) 
        values = (1/(df["price"]+1000))        # als=6.5
        
        #values = (1 / (df["price"]**0.1 + 1000)) # lmf=6.7?

        # Each user must be mapped to an uid, an integer between 0 and num_user-1 (Same for courses)
        uids = users.cat.codes
        iids = items.cat.codes
        self.user_item_matrix = csr_matrix((values, (uids, iids)))

        # The following two dicts store the mappings.
        self.user2uid = {cat: i for i, cat in enumerate(users.cat.categories)}
        self.iid2item = dict(enumerate(items.cat.categories))
        
    def evaluate(self, df: pd.DataFrame, top=50):
        users = df["user_id"]
        recommendations = self.recommend(users, top)
        score = mapk(df["course_id"], recommendations)
        logger.info(f"Score of evaluation: {score*100}")

    def recommend(self, users: List[Any], top=50) -> List[List[Any]]:
        '''
        Recommend each user `top` items.
        '''
        uids = [self.user2uid[u] for u in users]
        iids, _ = self.model.recommend(uids, self.user_item_matrix[uids], filter_already_liked_items=False, N=top)
        items = [[self.iid2item[i] for i in row] for row in iids]

        return items
    

def parse_args():
    parser = ArgumentParser()

    # input files
    parser.add_argument("--course_file", type=str, default="./data/courses.csv", help="File that contains information about courses")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training file")
    parser.add_argument("--eval_file", type=str, default=None, help="Path to the evaluation file")
    parser.add_argument("--test_file", type=str, default=None, help="Path to the test file")
    
    # output directory
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to be saved for recommendation output")

    # Model and training parameters
    parser.add_argument("--model_name", type=str, default="als", help="The name of model to be used. Option: 'als', 'bpr', 'lmf' ")    
    parser.add_argument("--factors", type=int, default=100, help="Dimension of latent factors")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iteration in als")
    parser.add_argument("--lr", type=float, default=1., help="Learning rate, used in bpr, lmf")
    parser.add_argument("--regularization", type=float, help="Regularization factor")

    # For lmf
    parser.add_argument("--neg_prop", type=int, default=30, help="Proportion of negative samples.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    paths = {
        "train": args.train_file,
        "eval": args.eval_file,
        "test": args.test_file,
    }

    logger.info("Preparing datasets and model...")
    datasets = Hahow_Dataset(paths, args.course_file)
    model = Implicit_Model(args)
    
    # Train
    logger.info("Training...")
    train_data = datasets["train"]
    model(train_data)

    # Evaluate
    logger.info("Evaluating...")
    eval_data = datasets["eval"]
    if eval_data is not None:
        model.evaluate(eval_data)

    # Test
    logger.info("Testing...")
    test_data = datasets["test"]
    if test_data is not None:
        recommendations = model.recommend(test_data["user_id"])
        
        with open(os.path.join(args.output_dir, "recommendations.csv"), 'w') as fp:
            fp.write("user_id,course_id\n")
            for user, items in zip(test_data["user_id"], recommendations):
                fp.write(user + "," + " ".join(items) + "\n")
    
    logger.info("ok")