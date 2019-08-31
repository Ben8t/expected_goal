import argparse
import pandas
from joblib import load
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="csv file containing shots data.")
    parser.add_argument("-s", "--save", help="input for saving file", action="store_true")
    args = parser.parse_args()

    pipeline = load("./data/model/xgboost_model.pkl")
    data = pandas.read_csv(args.data)
    
    y_pred = pipeline.predict_proba(data)
    data["expected_goal"] = [y[1] for y in y_pred]
    
    if args.save:
        file_path = input("File path (for example /Users/username/folder/file.csv): ")
    else:
        timestamp = datetime.datetime.strftime(datetime.datetime.utcnow(), "%Y%m%d%H%M%S")
        file_path = f"./data/artifacts/predict_data_{timestamp}.csv"
            
    data.to_csv(file_path, index=False)