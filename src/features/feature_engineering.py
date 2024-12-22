import pandas as pd 
import numpy as np 
import pathlib
from pathlib import Path
import yaml
from sklearn.feature_extraction.text import CountVectorizer



def target_path(path: str) -> str:
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path



def app_vectorizer(df: pd.DataFrame) -> pd.DataFrame:
    try: 
      df.fillna('hi', inplace=True)
    except AttributeError as e:
      print(f"Error: Input df is not valid dataframe {e}")
    
    try: 
      X_train = df['content'].values
      y_train = df['sentiment'].values
    except KeyError as e:
      print(f"column not found {e}")
    
    try: 
    # Apply Bag of Words (CountVectorizer)
      vectorizer = CountVectorizer()    
      # Fit the vectorizer on the training data and transform it
      X_bow = vectorizer.fit_transform(X_train)
    except ValueError as e:
      print(f"Error: issue with input data {e}")
    except Exception as e:
      print(f"Error: Unknown issue {e}")
    try: 
      final_df = pd.DataFrame(X_bow.toarray())
      final_df['label'] = y_train
    except Exception as e:
      print(f"Error: Unable to create a final dataframe {e}")
    else: 
      return final_df

def main():
    try: 
      input_path =  target_path("/data/processed/")
      train_data = pd.read_csv(input_path+"train_processed.csv")
      test_data = pd.read_csv(input_path+"test_processed.csv")  
    except FileNotFoundError as e:
      print(f"Erro: File not present at location {e}")
    except Exception as e:
      print(e)
      
    try:   
      train_bow = app_vectorizer(train_data)
      test_bow = app_vectorizer(test_data)
    except Exception as e:
      print(e)
      
    try: 
      save_path = target_path('/data/feature_eng/')
      train_bow.to_csv(save_path+'train_bow.csv')
      test_bow.to_csv(save_path+'test_bow.csv')
    except PermissionError:
      print(f"Error: Not allowed to write {e}")
    except FileNotFoundError as e:
      print(f"Error: {e}")
    except Exception as e:
      print(e)
if __name__ == '__main__':
    main()

