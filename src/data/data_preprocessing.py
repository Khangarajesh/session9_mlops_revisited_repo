import re 
import os
os.add_dll_directory(r"C:\Users\Rajesh Assignment\MLOPS\session9_mlops_revisited_repo\mlops_revisited_env\lib\site-packages\sklearn\.libs")
import sklearn
import nltk
import string 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import pathlib
import pandas as pd 
import numpy as np

nltk.download('wordnet')
nltk.download('stopwords')

def target_path(path):
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

def save_data(train_data_p, test_data_p):
    
    train_path = target_path('/data/processed/') + 'train_processed.csv'
    test_path = target_path('/data/processed/') + 'test_processed.csv'
    print(train_path, test_path)
    train_data_p.to_csv(train_path)
    test_data_p.to_csv(test_path)

def main():
    input_path =  target_path("/data/raw/")
    train_df = pd.read_csv(input_path+"train.csv")
    test_df = pd.read_csv(input_path+"test.csv")   

    train_process_df = normalize_text(train_df)
    test_process_df = normalize_text(test_df)

    save_data(train_process_df, test_process_df)
    
if __name__ == '__main__':

    main()

