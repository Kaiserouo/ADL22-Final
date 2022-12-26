# trick vscode syntax highlighter into highlighting functions in `utils.py`
# but still works if `python two-towels.py` instead of having to `cd .. & python -m src.two-towel`
try:
    from .utils import *
except:
    from utils import *

import torch
from torch import nn
import itertools
import pandas as pd
from tqdm import tqdm
import colorama
from torch.utils.data import DataLoader

import torch.nn.functional as F



class EncodingModel(torch.nn.Module):
    """
        autoencoder?
        turn input vector into some encoded form, should be used with other stuff...

        forward: (*, dim_input) -> (*, dim_output)
        the flow is dim_input -> dim_hidden [-> dim_hidden] -> dim_output
        with the middle `[]` part `n_hidden` times
    """
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, p_dropout=0.1):
        super().__init__()
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
                        nn.Linear(dim_hidden, dim_hidden),
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
        super().__init__()
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
        user_result = self.user_model(x['user_embed'])
        item_result = self.item_model(x['item_embed'])
        return self.cos_sim(user_result, item_result)

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

            weight: `1` if y == 1, `w` if y == 0
            then weight = (1 - w) * y + w

            do note that cosine similarity ranges between [-1, 1]
            so map y to [-1, 1] by (2y-1)
        """
        weight = (1 - w) * y + w
        result = torch.sum(torch.pow(x - (y * 2 - 1), 2) * weight)
        # print(f'x: {x}, y: {y}, result: {result}')
        return result
    return loss

def saveUserItemModel(args, tt_model, accelerator):
    os.makedirs(args.save_tt_model_dir, exist_ok=True)
    torch.save(
        accelerator.unwrap_model(tt_model).state_dict(), 
        os.path.join(args.save_tt_model_dir, 'tt_model.pt')
    )
def saveModelArguments(args, tt_model):
    os.makedirs(args.save_tt_model_dir, exist_ok=True)
    with open(os.path.join(args.save_tt_model_dir, 'args.txt'), 'w') as fout:
        fout.write(f'Args: {args}\n\n')
        fout.write(f'- tt_dim_hidden: {args.tt_dim_hidden}\n')
        fout.write(f'- tt_n_hidden: {args.tt_n_hidden}\n')
        fout.write(f'- tt_p_dropout: {args.tt_p_dropout}\n')
        fout.write(f'- tt_dim_encoding: {args.tt_dim_encoding}\n\n')
        fout.write(str(tt_model))
    
    
def loadUserItemModel(args):
    model = makeUserItemModel(args)
    model.load_state_dict(torch.load(
        os.path.join(args.load_tt_model_dir, 'tt_model.pt')
        )
    )
    return model

def mainUserItemTraining(args):
    """
        Main training script. 
        Uses ArgumentParser.ttModelArguments and ArgumentParser.trainingArguments so make sure
        their value is set.
    """


    accelerator = Accelerator(fp16=args.fp16)
    accelerator.print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')

    # load model & dataset, refer to feature_extract.py to know how those 2 come to be
    accelerator.print(f'{colorama.Fore.BLUE}[*] Preparing...{colorama.Fore.RESET}')
    eds = loadUserItemExamplesDataset(args)
    fdss = loadUserItemFeatureDataset(args)
    user_embed, user_map, item_embed, item_map = makeUserItemEmbedding(fdss)
    train_dataloader = makeUserItemDataloader(args, eds, user_embed, user_map, item_embed, item_map, is_train=True)

    tt_model = makeUserItemModel(args)
    if args.load_tt_model_dir is not None:
        accelerator.print(f'{colorama.Fore.BLUE}[*] Continuing from checkpoint...{colorama.Fore.RESET}')
        loadUserItemModel(args)
    loss_fn = makeUserItemLossFunction(args)
    optimizer = torch.optim.AdamW(tt_model.parameters(), lr=args.learning_rate)

    tt_model, optimizer, train_dataloader = accelerator.prepare(
        tt_model, optimizer, train_dataloader
    )

    saveModelArguments(args, tt_model)

    accelerator.print(f'{colorama.Fore.GREEN}[o] Preparation complete!{colorama.Fore.RESET}')
    accelerator.print(f'{colorama.Fore.BLUE}[*] Start training...{colorama.Fore.RESET}')

    # training
    # (for batch size 512 (542 it/ep): 8.5it/s, 10 epoch takes roughly 12min)
    progress_bar = tqdm(range(len(train_dataloader) * args.num_epochs), disable=not accelerator.is_main_process)
    dl_len = len(train_dataloader)
    for epoch in range(args.num_epochs):
        progress_bar.set_description(f'Epoch: {epoch+1}/{args.num_epochs}')
        tt_model.train()
        for i, batch in enumerate(train_dataloader):
            with accelerator.accumulate(tt_model):
                output = tt_model(batch)
                loss = loss_fn(output, batch['label'])
                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({'Step': f'{i}/{dl_len}', 'loss': loss.item()})
            
            # if accelerator.sync_gradients:
                # if args.record_steps is not None and complete_step % args.record_steps == 0:
                #     # record & save
                #     pass

        accelerator.print('[*] Saving model...')
        saveUserItemModel(args, tt_model, accelerator)
        accelerator.print(f'{colorama.Fore.GREEN}[o] Model saved to {args.save_tt_model_dir}!{colorama.Fore.RESET}')

def RS_Filtering(args):
    """
        Recommendation system: filtering
    """
    pass
    

def saveRecommendation(args, rec_map):
    """
        save recommendation into csv format
    """
    os.makedirs(args.save_rec_path, exist_ok=True)

    # get seen / unseen user list
    seen_df = pd.read_csv(args.test_item_seen_file)
    unseen_df = pd.read_csv(args.test_item_unseen_file)
    seen_user_ids = seen_df['user_id']
    unseen_user_ids = unseen_df['user_id']

    seen_rec_df = pd.DataFrame(
        [[user_id, ' '.join(rec_map[user_id][:50])] for user_id in seen_user_ids],
        columns=['user_id', 'course_id']
    )
    unseen_rec_df = pd.DataFrame(
        [[user_id, ' '.join(rec_map[user_id][:50])] for user_id in unseen_user_ids],
        columns=['user_id', 'course_id']
    )

    # save dataframe
    seen_rec_df.to_csv(os.path.join(args.save_rec_path, 'rec_seen_course.csv'), index=False)
    unseen_rec_df.to_csv(os.path.join(args.save_rec_path, 'rec_unseen_course.csv'), index=False)

def mainUserItemPredicting(args):
    accelerator = Accelerator(fp16=args.fp16)
    fdss = loadUserItemFeatureDataset(args)

    # get model
    tt_model = loadUserItemModel(args)
    tt_model.eval()
    user_model = tt_model.user_model
    item_model = tt_model.item_model

    # make item embedding
    fdss['item'].set_format(type="torch", columns=["embed"], output_all_columns=True)
    item_embed = item_model(fdss['item']['embed']).to(accelerator.device) # (728, 500)
    item_ids = fdss['item']['item_id']

    # A: (a_dim, dim), B: (b_dim, dim) -> (a_dim, b_dim)
    # cannot use nn.CosineSimilarity for this case
    cos_sim = lambda A, B: F.normalize(A) @ F.normalize(B).t()

    # rank
    accelerator.print(f'{colorama.Fore.BLUE}[*] Ranking on {args.load_tt_model_dir}...{colorama.Fore.RESET}')
    fdss['user'].set_format(type="torch", columns=["embed"], output_all_columns=True)
    dataloader = DataLoader(fdss['user'], batch_size=4096)
    rec_map = dict()    # user_id -> list[item_id]
    for batch in tqdm(dataloader, desc="Ranking...."):
        pred = cos_sim(user_model(batch['embed']).to(accelerator.device), item_embed)  # (2048, 728)
        _, indices = torch.sort(pred, dim=1, descending=True)
        # then `indices[uid]` will be the ranked iid
        for uid, user_id in enumerate(batch['user_id']):
            rec_map[user_id] = [item_ids[i] for i in indices[uid, :].tolist()]
    
    # filter
    
    # output
    saveRecommendation(args, rec_map)

if __name__ == '__main__':
    args = parse_args()
    if args.two_towel_task == 'train':
        mainUserItemTraining(args)
    if args.two_towel_task == 'predict':
        mainUserItemPredicting(args)