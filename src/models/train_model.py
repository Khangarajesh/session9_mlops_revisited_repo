

import pandas as pd 
import numpy as np 
import pathlib
from pathlib import Path
import yaml
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import pickle as pkl

def target_path(path: str) -> str:
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path



def model_training(df: pd.DataFrame) -> xgb.sklearn.XGBClassifier:
    try:
      X_train_bow = df.iloc[:,0:-1]
      y_train = df.iloc[:,-1]
      # Define and train the XGBoost model
      xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
      xgb_model.fit(X_train_bow, y_train)
    except ValueError as e:
      print(f"Error: {e}")
    except Exception as e:
      print(f"Error: {e}")
    return xgb_model

def save_model(model: xgb.sklearn.XGBClassifier) -> None:
    try: 
      model_path = target_path("/models/")
      pkl.dump(model,open(model_path+'model.pkl','wb'))
    except FileNotFoundError as e:
      print(f"Error: {e}")
    except PermissionError as e:
      print(f"Error: {e}")
    except Exception as e:
      print(f"Error: {e}")
    
def main():
    try:
      file_path = target_path('/data/feature_eng/')
      train_df = pd.read_csv(file_path+"train_bow.csv")
    except FileNotFoundError as e:
      print(f"Error: file not found {e}")
    except Exception as e:
      print(f"Error: Some unknown issue {e}")
    
    try: 
      model = model_training(train_df)
      save_model(model)
    except Exception as e:
      print(f"Error: Isuue occured while training model {e}")
    
if __name__ == "__main__":
    main()