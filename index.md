---
layout: default
---

## Distributional Fairness-aware Recommendation

## 1 Abstract

Fairness has been gradually recognized as a significant problem in the recommendation domain.
Previous models usually achieve fairness by reducing the average performance gap between different user groups. 
However, the average performance may not sufficiently represent all the characters of the performances in a user group.
Thus, equivalent average performance may not mean the recommender model is fair, for example, the variance of the performances can be different.
To alleviate this problem, in this paper, we define a novel type of fairness, where we require that the performance distributions across different user groups should be similar. 
We can prove that with the same performance distribution, the numerical characteristics of the group performance, including the expectation, variance and any high order moments are also the same. 
To achieve distributional fairness, we propose a generative and adversarial training framework. 
In specific, we regard the recommender model as the generator to compute the performance for each user in different groups, and then we deploy a discriminator to judge which group the performance is drawn from.
By iteratively optimizing the generator and discriminator, we can theoretically prove that the optimal generator (recommender model) can indeed lead to the equivalent performance distributions.
To smooth the adversarial training process, we propose a dual curriculum learning strategy to schedule the training samples for better model optimization. To make our framework more adaptive to the Top-N recommendation task, we soften the ranking metrics, and use them to measure the performance discrepancy.
We conduct extensive experiments based on real-world datasets to demonstrate the effectiveness of our model. 
For benefiting the research community, we have released our project at https://distfair.github.io.

## 2 Contributions

In a summary, the main contributions of this paper are as follows:
- We formally define a novel type of recommendation fairness, where we would like to reduce the performance distribution gap across different user groups.
- To achieve distributional fairness, we design an adversarial model, and propose a dual curriculum learning strategy to smooth its training process. 
To better optimize the ranking metrics, we design a soft strategy to make our framework fully differentiable.
- Theoretically, we prove that (\romannumeral1) our adversarial method can indeed lead to the equivalent performance distributions and (\romannumeral2) with the same performance distributions, the performance expectation, variance and any higher order moment are also the same.
- Empirically, we conduct extensive experiments based on four real-world datasets to demonstrate the effectiveness of our framework. To promote this research direction, we have released our project at https://distfair.github.io.

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

| Hyper-parameter     | Explanation | Range |
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

As different base models have different hyper-paramerters to tune, you can view the details in the corresponding model files.