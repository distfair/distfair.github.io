# Meta Recommendation with Robustness Improvement

## 1 Abstract

Meta-learning has been recognized as an effective remedy for solving the cold-start problem in the recommendation domain.
Traditional models assume that the testing samples are always distributionally aligned with the training ones. However, in the cold start setting, we can only observe a small number of users and items, which, in practice, may fail to represent the newly arrived (testing) sample distributions, and thus lead to lowed recommendation performance. For alleviating this problem, in this paper, we propose a robust meta recommender framework to address the distribution shift problem. In specific, we argue that the distribution shift may exist on both of the user- and item-levels, and in order to remove them simultaneously, we design a novel distributionally robust model by hierarchically reweighing the training samples. Generally speaking, the sample weights are leveraged to tune the training distribution, and we minimize the worst-case loss by searching the weights on a unit ball, which is expected to improve the robustness of the learned model. Theoretically, we analyze the convergence rate and demonstrate the generalization capability of our framework. Empirically, we conduct extensive experiments based on different meta recommender models and real-world datasets to verify the generality and effectiveness of our framework. For benefiting the research community and promoting this direction, we have released our code at this page.

## 2 Contributions

In a summary, the main contributions of this paper can be concluded as follows:

- We propose to improve the robustness of the meta recommender models for alleviating the distribution shift problem, which, to our knowledge, is the first time in the recommendation domain. 

- To achieve the above idea, we design a hierarchical reweighing mechanism to remove the distribution shifts on the user- and item-level simultaneously. In addition, we provide theoretical foundations and insights for the proposed framework.

- We conduct extensive experiments based on real-world datasets to demonstrate the effectiveness and generality of our framework, and for promoting this direction, we have released our project.

## 3 Dataset Overview

| Dataset        | # User | # Item | # Interaction | Sparsity | Domain |
| -------------- | ------ | ------ | ------------- | -------- | ------ |
| MovieLens-100K | 943    | 1,682  | 100,000       | 93.70%   | Movies |
| MovieLens-1M   | 6,040  | 3,883  | 1,000,000     | 95.74%   | Movies |
| Anime          | 20,000 | 9,499  | 1,771,114     | 99.07%   | Animes |
| Jester         | 24,983 | 100    | 1,082,498     | 56.67%   | Jokes  |
| Steam          | 20,000 | 14,430 | 295,485       | 99.90%   | Games  |



## 4 Quick Start

### Step 1: Download the project

First of all, download our project `RobMeta.zip` from [Google Drive](https://drive.google.com/file/d/1Uh72K-T7oU4--wlaDRneeUIKnKHhvBtA/view?usp=sharing) and unzip the file. The file includes both codes and datasets.

### Step 2: Create the running environment

Create `Python 3.8` enviroment and install the packages that the project requires.
- numpy==1.22.3
- PyYAML==6.0 
- scikit_learn==1.1.2 
- torch==1.11.0

You can install the packages with the following command.

```
    pip install -r requirements.txt
```

### Step 3: Run the project

Choose a dataset to run (e.g. MovieLens-1M) with the following command.

```
    cd movielens-1m
```

Choose a model (e.g. MeLU) to run with the following command.

```
    python run.py MeLU hiddenDim1 32 hiddenDim2 32 localLr 0.1 lr 0.1 gamma 0.99 batchSize 20 epoch 200 userNum 20000
```

You can also use `quick_start.py` to run the project conveniently.

```
    python quick_start.py
```

You can also change the hyper-parameters as you want. The necessary hyper-parameters for each model have been recorded in the `hyperParam` dictionary.

### Step 4: Check the performance

The performance will be saved in `performance.csv`. The first column is the name of models. The second column is the number of testing sets (with different β). The third to the sixth columns are metrices `NDCG@10, Recall@10, Precision@10, F1@10` in order.


## 5 Hyper-parameters search range

We tune hyper-parameters according to the following table.

| Hyper-parameter     | Explain | Range |
| ------------------- | ---------------------------------------------------- | ------------------- |
| taskWeightLr | user weights lr | \{0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1\} |
| sampleWeightLr | interaction weights lr | \{0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1\} |
| localLr     | adapt lr | \{0.001, 0.01, 0.05, 0.1\} |
| lr   | model lr |  \{0.001, 0.01, 0.05, 0.1\} |
| batchSize | batch size |  \{20, 100, 200, 400\} |
| gamma | learning rate decay | \{0.95, 0.97, 0.99\} |
| hiddenDim1 | hidden layer1 dim | \{16, 32, 64, 128\} |
| hiddenDim2 | hidden layer2 dim | \{16, 32, 64, 128\} |
| indHiddenDim | index embedding dim | \{16, 32, 64, 128\} |

As different base models have different hyper-paramerters to tune, you can view the details in corresponding model files.orresponding model files.