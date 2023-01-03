# NTU ADL 2022-fall final project

+ It is a implement of course recommendation system. Our model is base on behaviour sequence transformers model.
    + our target is to find a function S to predict the last position of a user's behaviour sequence, which is the possibility of u's next behaviour is buy the traget item v.
    + $S(u)=\{v_1,v_2,v_3,...,v_n\}$
    + reference:Chen, Q., Zhao, H., Li, W., Huang, P., & Ou, W. (2019, August). Behavior sequence transformer for e-commerce recommendation in alibaba. In Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data (pp. 1-4).
    + reference code: https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/movielens_recommendations_transformers.ipynb
+ Training data is from a Taiwan online course website "hahow" consist of three part:
    + user context: user_id, interest, gender, occupation
    + course context: course_id, course name, genre
    + rating data: user_id, course_id, rating, the time user buy the course

# reproduce
```shell
# download raw data (hahow user data)
bash download.sh
# train model and do the predict, it takes about 5 min in colab environment(with gpu)
bash run.sh path/to/pred.csv
```

