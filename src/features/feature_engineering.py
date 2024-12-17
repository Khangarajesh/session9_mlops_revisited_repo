import pandas as pd 
import numpy as np 
import pathlib
from pathlib import Path
import yaml
from sklearn.feature_extraction.text import CountVectorizer



def target_path(path):
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path



def app_vectorizer(df):
    df.fillna('hi', inplace=True)
    X_train = df['content'].values
    y_train = df['sentiment'].values
    # Apply Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer()    
    # Fit the vectorizer on the training data and transform it
    X_bow = vectorizer.fit_transform(X_train)
    final_df = pd.DataFrame(X_bow.toarray())
    final_df['label'] = y_train
    return final_df

def main():
    input_path =  target_path("/data/processed/")
    train_data = pd.read_csv(input_path+"train_processed.csv")
    test_data = pd.read_csv(input_path+"test_processed.csv")  
    
    train_bow = app_vectorizer(train_data)
    test_bow = app_vectorizer(test_data)
    
    save_path = target_path('/data/feature_eng/')
    train_bow.to_csv(save_path+'train_bow.csv')
    test_bow.to_csv(save_path+'test_bow.csv')
    
if __name__ == '__main__':
    main()

