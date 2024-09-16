import sys
import os
import numpy as np
from src.exception import CustomException
from src.utils import load_object
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, text):
        try:
            model_path = os.path.join("artifacts", "text_classifier_model.h5")
            tokenizer_path = os.path.join("artifacts", "tokenizer.pkl")
            label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            tokenizer = load_object(file_path=tokenizer_path)
            label_encoder = load_object(file_path=label_encoder_path)
            print("After Loading")

            
            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=500)

            
            preds = model.predict(padded)
            predicted_category = label_encoder.inverse_transform([np.argmax(preds)])
            
            return predicted_category[0]

        except Exception as e:
            raise CustomException(e, sys)


