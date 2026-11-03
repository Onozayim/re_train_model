import re
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from re_train_helper import load_corrected_emails

MODEL_PATH = "emotion_model.keras"

TOKENIZER_PATH = "tokenizer.pkl"
# LABEL_ENCODER_PATH = "label_encoder.pkl"

ORIGINAL_DATASET = "findal_dataset.csv"

OUTPUT_MODEL = "emotion_model" + " 2.keras"


df_original = pd.read_csv("findal_dataset.csv")
df_corrected = load_corrected_emails()

if len(df_corrected) == 0:
    sys.exit("no new corrected_emails, we end it here")

df = pd.concat([df_original, df_corrected], ignore_index=True)
# df = df_original
df = df.drop_duplicates()


remove_words = ["name"]

# Create regex pattern with word boundaries
pat = r"\b(?:{})\b".format("|".join(remove_words))

# Remove words ignoring the case
df["text"] = df["text"].str.replace(pat, "", regex=True, flags=re.IGNORECASE)

# Step 1: Remove URLs
df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)

# Step 2: Remove special characters and punctuation
df["text"] = df["text"].str.replace(r"[^\w\s]", "", regex=True)

# Step 3: Remove extra whitespaces
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

# Step : 4 Remove numeric values
df["text"] = df["text"].str.replace(r"\d+", "", regex=True)

# Step 6: Lowercasing
df["text"] = df["text"].str.lower()

# Step 8: Remove stop words
stop = stopwords.words("english")
df["text"] = df["text"].apply(
    lambda x: " ".join([word for word in x.split() if word not in (stop)])
)

# Step : 9 Remove non-alphanumeric characters from the 'Text' column
df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))


# # Encode labels
# df['emotion'] = df['emotion'].replace('Sadness',0)
# df['emotion'] = df['emotion'].replace('Joy',1)
# df['emotion'] = df['emotion'].replace('Disgust',2)
# df['emotion'] = df['emotion'].replace('Anger',3)
# df['emotion'] = df['emotion'].replace('Fear',4)

print(df["emotion"].unique())
print(df['emotion'].dtype)

# df["emotion"] = df["emotion"].str.strip().str.lower()

# df["emotion"] = df["emotion"].str.strip().str.lower()

emotion_map = {
    "sadness": 0,
    "joy": 1,
    "disgust": 2,
    "anger": 3,
    "fear": 4
}

# df["emotion"] = df["emotion"].map(emotion_map)
# df['emotion'] = pd.to_numeric(df['emotion']).astype(np.int32)

print(df['emotion'].unique())
print(df['emotion'].dtype)

print(df["emotion"].isnull().sum())

# Download NLTK resources (uncomment the following line if not already downloaded)?
nltk.download('punkt')
nltk.download('stopwords')

X = df['text']
y = df['emotion']

# Train Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.dtype)
print(y_test.dtype)
print(df["emotion"].unique())
# Tokenize the text data
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Max Len in X_train_sequences
maxlen = max(len(tokens) for tokens in X_train_sequences)

# Perform padding on X_train and X_test sequences
X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=maxlen,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=maxlen,
    padding="post",
    truncating="post"
)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

# Embedding Input Size / Vocabulary Size 
vocab_size = min(len(tokenizer.word_index) + 1, 50000)

# Define the model
model = Sequential()

# Add embedding layer
model.add(Embedding(
    vocab_size,
    100,
    trainable=True
))         

# Dropout
model.add(Dropout(0.5))

# Add Bidirectional LSTM layer
model.add(Bidirectional(GRU(120, return_sequences=True)))
model.add(Bidirectional(GRU(64, return_sequences=True)))

#Batch Normalization
model.add(BatchNormalization())

# Add Bidirectional GRU layer
model.add(Bidirectional(GRU(64)))

# Add output layer
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Model Train 
history = model.fit(X_train_padded, y_train, epochs=20, batch_size=1500, validation_data=(X_test_padded, y_test))

model.save(OUTPUT_MODEL)