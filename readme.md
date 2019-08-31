# Expected goals

Expected goals (xG) is the new revolutionary football metric, which allows you to evaluate team and player performance. 
In a low-scoring game such as football, final match score does not provide a clear picture of performance.
This is why more and more sports analytics turn to the advanced models like xG, which is a statistical measure of the quality of chances created and conceded.

Goal here is to create a simple model for shot quality evaluation. It can be improve with more data and more modelling.

---

## Build environnement

`docker build -t expected_goal .`

## Compute expected goals

`docker run --rm -it -v $(pwd):/app expected_goal python main.py data/data.csv -s`

## Train model

`docker run --rm -it -v $(pwd):/app expected_goal python train.py`