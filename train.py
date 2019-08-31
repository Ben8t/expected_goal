import numpy
import pandas
import xgboost as xgb
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from src.utils import split_dataset, evaluation_metrics, print_evaluation_metrics
from joblib import dump

if __name__ == "__main__":
    data = pandas.read_csv("./data/train/shot_data.csv")

    x_train, y_train, x_test, y_test = split_dataset(data, 0.2, "is_goal")


    numeric_features = ["minute", "second", "x_shot", "y_shot", "goal_distance"]
    categorical_features = ["previous_type_name"]
    feature_engineering = DataFrameMapper([
        (numeric_features, StandardScaler()),
        (categorical_features[0], LabelBinarizer())])

    pipeline = Pipeline(steps=[
            ("feature_engineering", feature_engineering),
            ("model", xgb.XGBClassifier(n_estimators=100, scale_pos_weight=9, max_depth=10, random_state=42))])

    pipeline.fit(x_train[numeric_features + categorical_features], y_train)

    y_pred = pipeline.predict(x_test[numeric_features + categorical_features])
    y_pred_proba = pipeline.predict_proba(x_test[numeric_features + categorical_features])

    metrics = evaluation_metrics(y_test, y_pred)
    print_evaluation_metrics(metrics)

    dump(pipeline, "./data/model/xgboost_model.pkl")