from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def split_dataset(data, split_rate, output_variable):
    """Split dataset in 4 parts : x_train, y_train, x_test, y_test.
    
    Args:
        data (pandas.Dataframe): dataframe containg shot data.
        split_rate (float): split rate to divide train and test set (often 0.3).
        output_variable (str): target to predict.
        
    Returns:
        (pandas.Dataframe, pandas.Dataframe, pandas.Dataframe, pandas.Dataframe): train and test datasets.
    """
    train_set, test_set = train_test_split(data, test_size=split_rate, random_state=0)
    x_train = train_set.drop(output_variable, axis=1)
    y_train = train_set[output_variable]
    x_test = test_set.drop(output_variable, axis=1)
    y_test = test_set[output_variable]
    return x_train, y_train, x_test, y_test

def evaluation_metrics(y_test, y_pred):
    """
    Compute metrics
    """
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    return {"f1": f1, "accuracy": accuracy, "roc_auc_score": roc}

def print_evaluation_metrics(metrics):
    for key, value in metrics.items():
        print(key.replace("_", " ").upper(), " : ", value)