import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('punkt')


df = pd.read_csv('BBC_News_processed.csv')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


df['ProcessedText'] = df['Text'].apply(preprocess_text)


label_encoder = LabelEncoder()
df['Category_target'] = label_encoder.fit_transform(df['Category'])


tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(df['ProcessedText'])
X = tokenizer.texts_to_sequences(df['ProcessedText'])
X = pad_sequences(X, maxlen=500)


y = to_categorical(df['Category_target'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)


model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(64))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(5, activation='softmax'))  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

model.save('text_classifier_model4.h5')
import pickle
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
