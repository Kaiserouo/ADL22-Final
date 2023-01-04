# Collaborative Filtering (CF)

Implements recommender system using collaborative filtering for the hahow dataset.

## Learning Algorithms for implicit CF
+ [Alternating Least Square (ALS)](http://yifanhu.net/PUB/cf.pdf)
+ [Bayesian Personalized Ranking (BPR)](https://arxiv.org/pdf/1205.2618.pdf)
+ [Logistic Matrix Factorization (LMF)](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)

## Dependency
+ python=3.9.15
+ pandas=1.5.2
+ scipy=1.9.3
+ implicit=0.5.2

## Usage
If `--eval_file` is specified, the result of mapk@50 will be printed.

If `--test_file` is specified, a recommendation file for user in the test file named `recommendation.csv` will be generated. 

### ALS
```
python run.py \
--train_file /path/to/train.csv \
--eval_file /path/to/seen_course.csv \
--test_file /path/to/test.csv \
--model_name als \
--factors 1 \
--iterations 40
```

### BPR
```
python run.py \
--train_file /path/to/train.csv \
--eval_file /path/to/seen_course.csv \
--test_file /path/to/test.csv \
--model_name bpr \
--lr 0.00005 \
--factors 128 \
--iterations 50
```

### LMF
```
python run.py \
--train_file /path/to/train.csv \
--eval_file /path/to/seen_course.csv \
--test_file /path/to/test.csv \
--model_name lmf \
--factors 10 \
--iterations 40 \
--lr 0.5
```

