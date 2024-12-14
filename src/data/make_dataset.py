import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import pathlib
from pathlib import Path

'''
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')


# delete tweet id
df.drop(columns=['tweet_id'],inplace=True)
final_df = df[df['sentiment'].isin(['happiness','sadness'])]

final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)



train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)



train_data.to_csv(save_path)
'''


def target_path(file_name):
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + file_name
    return target_file_path
    
