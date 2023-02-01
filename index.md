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
For benefiting the research community, we have released our project at [this page](https://distfair.github.io).

## 2 Contributions

In a summary, the main contributions of this paper are as follows:
- We formally define a novel type of recommendation fairness, where we would like to reduce the performance distribution gap across different user groups.
- To achieve distributional fairness, we design an adversarial model, and propose a dual curriculum learning strategy to smooth its training process. 
To better optimize the ranking metrics, we design a soft strategy to make our framework fully differentiable.
- Theoretically, we prove that (1) our adversarial method can indeed lead to the equivalent performance distributions and (2) with the same performance distributions, the performance expectation, variance and any higher order moment are also the same.
- Empirically, we conduct extensive experiments based on four real-world datasets to demonstrate the effectiveness of our framework. To promote this research direction, we have released our project at this page.
## 3 Dataset Overview

| Dataset        | ML-1M | CiteULike | Sports | Home |
| -------------- | ------ | ------ | ------------- | -------- |
| # User | 6,037    | 5,551  | 22,685       | 41,253   |
| # Item   | 3,952  | 16,979  | 12,300     | 19,364   |
| # Interaction  | 575,281 | 204,986  | 185,718     | 338,376  |
| Sparsity         | 97.59% | 99.78%    | 99.93%     | 99.96%  |
| Domain          | Movies | Papers | E-commerce       | E-commerce  |



## 4 Quick Start

### Step 1: Download the project

Our project `DistFair.zip` is available at [Google Drive](https://drive.google.com/file/d/1JPkAw2QvsDjMxrfe8Av1JVY_q76SHh0z/view?usp=sharing). Download and unzip our project. It contains both codes and datasets.
### Step 2: Create the running environment

Create `Python 3.9` enviroment and install the packages that the project requires.
- numpy==1.23.2
- scikit_learn==1.1.2 
- torch==1.12.1
- pybind11==2.10.0
- tkinter==0.1.0

You can install the packages with the following command. `pybind11` is necessary in our code to accelerate the sampling process.

```
    pip install -r requirements.txt
```

### Step 3: Run the project
Run our frameworks with the following command:
```
    cd ./code
    python main.py --dataset ml-1m --model mf --methods distfair --rec_lr 0.01
```
where 
- `--dataset`: dataset chosen from `['ml-1m', 'citeulike', 'sports', 'home]`;
- `--model`: base model chosen from `['mf', 'lgn', 'nmf']`;
- `--methods`: fairness chosen from `['value', 'non_parity', 'distfair']`;

### Step 4: Check the performance

For the recommendation performance, we use NDCG@10, Recall@10, Precision@10, F1@10 as evaluation metrics. For the fairness performance, we use $\Delta E, \Delta V$, KL as evaluation metrics. The results are reported in Table 3 in the original paper.


## 5 Hyper-parameters search range

We tune hyper-parameters according to the following table.

| Hyper-parameter     | Explanation | Range |
| ------------------- | ---------------------------------------------------- | ------------------- |
| rec_lr | learning rate of recommender model | \{0.0001, 0.001, 0.01, 0.1\} |
| disc_lr | learning rate of discriminator | \{0.0001, 0.001, 0.01, 0.1\} |
| decay | weight of l2-norm regularizer | \{0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1\} |
| bpr_batch_size | batch size |  \{256, 512, 1024, 2048\} |
| rec_dim    | embedding size | \{4, 8, 16, 32, 64\} |
| disc_dim | hidden size of discriminator | \{4, 8, 16, 32, 64\} |
| weight_disc   | weight of fairness constraints |  \{0.0001, 0.001, 0.01, 0.1, 1, 10\} |
| cl_speed | changing speeds of curriculum learning | \{1, 2, 3, 4, 5\} |
| tau | temprature parameter of soft ranker | \{0.0001, 0.001, 0.01, 0.1, 1, 10, 100\} |
