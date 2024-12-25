import pathlib
import pandas as pd 
import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
#path 
import logging
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def target_path(path: str) -> str:
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    print(target_file_path)
    #pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path

#load data 
def load_data(path_inp: str) -> tuple:
    
    file_dir = target_path(path_inp)
    df = pd.read_csv(file_dir)
    logger.debug("df crwated")
    x = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values
    return x, y 


#load model 
def model_load() -> xgb.sklearn.XGBClassifier:
    model_dir_path = target_path('/models/')
    model = pickle.load(open(model_dir_path+'model.pkl','rb'))
    return model

#model matricks 
def main():
    x_train,y_train = load_data('/data/feature_eng/train_bow1.csv')
    x_test,y_test = load_data('/data/feature_eng/test_bow1.csv')
    model = model_load()
    logger.debug("model loaded from pickle file")
    try:
      #y_pred = model.predict(x_test)
      y_pred = y_test
      logger.debug("prediction doone")
      accuracy = accuracy_score(y_test, y_pred)
      logger.debug(accuracy)
    except Exception as e:
      logger.error(e.with_traceback)
    
    #classification_rep = classification_report(y_test, y_pred

#main 
if __name__ == '__main__':
    main()