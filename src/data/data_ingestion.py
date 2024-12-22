import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import pathlib
from pathlib import Path
import yaml
import urllib.request
from urllib.request import URLError, HTTPError

def target_path(path: str) -> str:
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path

    
def load_params(params_path: str) -> int:
    try: 
      test_size = yaml.safe_load(open(params_path, 'r'))['data_ingestion']['test_size']
    except yaml.YAMLError as e:
      print(f"yaml error {e}") 
    except FileNotFoundError as e:
      print(f"file not found {e}") 
    except Exception as e:
      print(f"unknown error {e}")
    else:
      return test_size

def read_data(url: str) -> pd.DataFrame:
    try:
      df = pd.read_csv(url)
    except FileNotFoundError as e:
      print(f"file not found {e}") 
    except Exception as e:
      print(f"unknown error {e}") 
    else:   
      return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try: 
      df.drop(columns=['tweet_id'],inplace=True)
      final_df = df[df['sentiment'].isin(['happiness','sadness'])]
      final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
    except KeyError as e:
      print(f"column not found in dataframe {e}")
    except Exception as e:
      print(f"Unknown error found {e}")
    else:    
      return final_df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try: 
      train_path = target_path('/data/raw/') + 'train.csv'
      test_path = target_path('/data/raw/') + 'test.csv'
      train_data.to_csv(train_path)
      test_data.to_csv(test_path)
    except FileNotFoundError as e:
      print(f"File not found at path {e}")
    except FileExistsError as e:
      print(f"File not found {e}")
    except Exception as e:
      print(f"some unknown error occured")


def main():
    try: 
      test_size = load_params('params.yaml')
      df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
      final_df = process_data(df)
      train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    except yaml.YAMLError as e:
      print(f"yaml file not founs {e}")
    except HTTPError as e:
      print(f"HttpError occured {e.reason}")
    except URLError as e:
      print(f"Url error {e.reason}")
    except Exception as e:
      print("some error occured")
    else:
      save_data(train_data,test_data)

if __name__ == '__main__':
    main()