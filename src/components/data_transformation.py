import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    tokenizer_file_path: str = os.path.join('artifacts', "tokenizer.pkl")
    label_encoder_file_path: str = os.path.join('artifacts', "label_encoder.pkl")
    max_len: int = 500  
    max_words: int = 5000  

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data.")

            
            tokenizer = Tokenizer(num_words=self.data_transformation_config.max_words, lower=True)
            tokenizer.fit_on_texts(train_df['ProcessedText'])

            
            train_sequences = tokenizer.texts_to_sequences(train_df['ProcessedText'])
            train_padded = pad_sequences(train_sequences, maxlen=self.data_transformation_config.max_len)

            test_sequences = tokenizer.texts_to_sequences(test_df['ProcessedText'])
            test_padded = pad_sequences(test_sequences, maxlen=self.data_transformation_config.max_len)

            logging.info("Text tokenization and padding completed.")

            
            label_encoder = LabelEncoder()
            train_labels = label_encoder.fit_transform(train_df['Category'])
            test_labels = label_encoder.transform(test_df['Category'])

            logging.info("Label encoding completed.")

            
            save_object(self.data_transformation_config.tokenizer_file_path, tokenizer)
            save_object(self.data_transformation_config.label_encoder_file_path, label_encoder)

            return (
                np.array(train_padded),  
                np.array(train_labels),  
                np.array(test_padded),   
                np.array(test_labels),   
                self.data_transformation_config.tokenizer_file_path,  
                self.data_transformation_config.label_encoder_file_path  
            )

        except Exception as e:
            raise CustomException(e, sys)
