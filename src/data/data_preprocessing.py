import re 
import os
import nltk
import string 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import pathlib
import pandas as pd 
import numpy as np
import logging 

nltk.download('wordnet')
nltk.download('stopwords')

logger = logging.getLogger("data_preprocessing")
logger.setLevel('DEBUG')

consol_handler = logging.StreamHandler()
consol_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def target_path(path: str) -> str:
    '''This function returns the file path'''
    current_dir = pathlib.Path(__file__)
    home_dir = current_dir.parent.parent.parent
    target_file_path = home_dir.as_posix() + path
    pathlib.Path(target_file_path).mkdir(parents = True, exist_ok = True)
    return target_file_path

def lemmatization(text: str) -> str:
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text: str) -> str:
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text: str) -> str:

    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def removing_punctuations(text: str) -> str:
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try: 
      df.content=df.content.apply(lambda content : lower_case(content))
      logger.debug("letter made lower case")
      df.content=df.content.apply(lambda content : remove_stop_words(content))
      logger.debug("stop words removed")
      df.content=df.content.apply(lambda content : removing_numbers(content))
      logger.debug("numbers removed")
      df.content=df.content.apply(lambda content : removing_punctuations(content))
      logger.debug("removed punctuation")
      df.content=df.content.apply(lambda content : removing_urls(content))
      logger.debug("remove urls")
      df.content=df.content.apply(lambda content : lemmatization(content))
      logger.debug("applied lemetization")
    except Exception as e:
      logger.error(f"Error: {e}")
    else:
      return df

def save_data(train_data_p: pd.DataFrame, test_data_p: pd.DataFrame) -> None:
    
    train_path = target_path('/data/processed/') + 'train_processed2.csv'
    test_path = target_path('/data/processed/') + 'test_processed2.csv'
    try:
      train_data_p.to_csv(train_path)
      test_data_p.to_csv(test_path)
    except FileNotFoundError as e:
      logger.error(f"Error: File not present at path {e}")
    except Exception as e:
      logger.error(f"Error: {e}")

def main():
    try: 
      input_path =  target_path("/data/raw/")
      train_df = pd.read_csv(input_path+"train.csv")
      test_df = pd.read_csv(input_path+"test.csv")   
    except FileNotFoundError as e:
      logger.error(f"Error: File not present at path {e}")
    except Exception as e:
      logger.error(f"Error: some unknown issue {e}")
      
    try:   
      train_process_df = normalize_text(train_df)
      test_process_df = normalize_text(test_df)
      logger.info("normalization done")
    except Exception as e:
      logger.error(f"Error: While returning normalize_text function output {e}")
    else:
      save_data(train_process_df, test_process_df)
      logger.info("files saved in data/processed/")
    
if __name__ == '__main__':

    main()

