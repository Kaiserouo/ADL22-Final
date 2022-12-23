from utils import *

import torch
from torch import nn
import itertools
import pandas as pd

class EncodingModel(torch.nn.Module):
    """
        autoencoder?
        turn input vector into some encoded form, should be used with other stuff...

        forward: (*, dim_input) -> (*, dim_output)
        the flow is dim_input -> dim_hidden [-> dim_hidden] -> dim_output
        with the middle `[]` part `n_hidden` times
    """
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, p_dropout=0.1):
        self.model = nn.Sequential(
            # (*, dim_input) -> (*, dim_hidden)
            nn.Linear(dim_input, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.Dropout(p=p_dropout),
            nn.LeakyReLU(),

            # (*, dim_hidden) -> (*, dim_hidden)
            *itertools.chain.from_iterable(
                [
                    [
                        nn.Linear(dim_input, dim_hidden),
                        nn.BatchNorm1d(dim_hidden),
                        nn.Dropout(p=p_dropout),
                        nn.LeakyReLU()
                    ]
                    for _ in range(n_hidden)
                ]
            ),

            # (*, dim_hidden) -> (*, dim_output)
            nn.Linear(dim_hidden, dim_output)
        )
    def forward(self, x):
        return self.model(x)

class TwoTowelModel(torch.nn.Module):
    """
        Two towel model: input user and item feature vector into encoded form, and use their
        cosine similarity as probability that a user likes an item

        forward: (*, dim_user), (*, dim_item) -> (*)
    """
    def __init__(self, user_model, item_model):
        """
            user_model: (*, dim_user) -> (*, dim_encoding)
            item_model: (*, dim_item) -> (*, dim_encoding)
            import by yourself, MAKE SURE THEY MAP TO SAME DIMENSION! (dim_encoding)
        """
        self.user_model = user_model
        self.item_model = item_model
        self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x):
        """
            x: dict, at least contain: {
                'user_embed': (*, dim_user)
                'item_embed': (*, dim_item)
            }
            will return their cosine similarity, shape (*)
                        (*, dim_user), (*, dim_item) 
            -[model]->  (*, dim_encoding), (*, dim_encoding)
            -[cosine]-> (*)
        """
        self.user_result = self.user_model(x['user_embed'])
        self.item_result = self.item_model(x['item_embed'])
        return self.cos_sim(self.user_result, self.item_result)

def makeUserItemModel(args):
    # do note that the input dimension depends on the feature vector generating process
    # (i.e. makeUserItemFeatureDataset stuff)
    # I am hard-coding this in but do change this if you change the feature vector dimensions
    # (user: 772, item: 860)

    user_model = EncodingModel(772, args.tt_dim_encoding, args.tt_dim_hidden, args.tt_n_hidden, args.tt_p_dropout)
    item_model = EncodingModel(860, args.tt_dim_encoding, args.tt_dim_hidden, args.tt_n_hidden, args.tt_p_dropout)
    return TwoTowelModel(user_model, item_model)

def makeUserItemLossFunction(args):
    """
        return a loss function for user-item.
        I use Weighted Matrix Factorization loss function:

        loss(x) = SquareError(x[label == 1]) + w * SquareError(x[label == 0])
        
        where `w` is the weight of negative samples.
        here, negative sampling is effectively sampled uniformly by making the
        user-item example dataset containing negative samples

        ref. https://developers.google.com/machine-learning/recommendation/collaborative/matrix
    """
    w = args.tt_ns_w
    def loss(x, y):
        """
            x: result, should be cosine similarity, shape (*)
            y: label, should contain only 0 and 1, shape (*)
        """
        return 
        
def mainUserItemTraining(args):
    accelerator = Accelerator(fp16=args.fp16)
    accelerator.print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')

    # load model & dataset

    # example dataset
    # dss = loadDataset(args)
    # pdss = preprocessDataset(args, dss)
    # df = pd.read_csv('../hahow/data/train.csv')
    # eds = makeUserItemExamplesDataset(dss, df)
    # eds.save_to_disk('eds.hf')
    eds = loadUserItemExamplesDataset(args)

    # fdss = makeFeatureDataset(args, accelerator, pdss)
    fdss = loadUserItemFeatureDataset(args)
    user_embed, user_map, item_embed, item_map = makeUserItemEmbedding(fdss)
    dataloader = makeUserItemDataloader(args, eds, user_embed, user_map, item_embed, item_map, is_train=True)

    tt_model = makeUserItemModel(args)

    # TODO: inject parameters in (by making makeOptimizer?)
    optimizer = torch.optim.AdamW(tt_model.parameters())

    tt_model, optimizer = accelerator.prepare(
        tt_model, optimizer
    )


if __name__ == '__main__':
    args = parse_args()
    mainUserItemTraining(args)
    