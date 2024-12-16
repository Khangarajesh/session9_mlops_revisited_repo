import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import pathlib
from pathlib import Path
import yaml


def target_path(path):
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path

    
def load_params(params_path) :
    test_size = yaml.safe_load(open(params_path, 'r'))['data_ingestion']['test_size']
    return test_size

def read_data(url):
    df = pd.read_csv(url)
    return df

def process_data(df):

    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]
    final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
    return final_df

def save_data(train_data, test_data):
    
    train_path = target_path('/data/raw/') + 'train.csv'
    test_path = target_path('/data/raw/') + 'test.csv'
    print(train_path, test_path)
    train_data.to_csv(train_path)
    test_data.to_csv(test_path)



def main():
    test_size = load_params('params.yaml')
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = process_data(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

    save_data(train_data,test_data)

if __name__ == '__main__':
    main()