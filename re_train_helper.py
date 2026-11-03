import os
import re

import nltk
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")

load_dotenv()

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

# MySQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", '3306'))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "emailanalyzer")

def lower_case(text):
    text = str(text)

    text = text.split()

    text = [
        word.lower()
        for word in text
    ]

    return " ".join(text)


def remove_stop_words(text):

    text = str(text)

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


def removing_numbers(text):

    text = str(text)

    return "".join(
        char
        for char in text
        if not char.isdigit()
    )


def removing_punctuations(text):

    text = re.sub(
        '[%s]' % re.escape(
            """!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""
        ),
        " ",
        str(text)
    )

    text = text.replace("؛", "")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return " ".join(text.split()).strip()


def removing_urls(text):

    url_pattern = re.compile(
        r"https?://\S+|www\.\S+"
    )

    return url_pattern.sub(
        "",
        str(text)
    )


def lemmatization(text):

    text = str(text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def normalized_sentence(sentence):

    sentence = lower_case(sentence)

    sentence = remove_stop_words(sentence)

    sentence = removing_numbers(sentence)

    sentence = removing_punctuations(sentence)

    sentence = removing_urls(sentence)

    sentence = lemmatization(sentence)

    return sentence

def get_database_connection():

    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )


# ============================================================
# LOAD CORRECTED EMAILS
# ============================================================

def load_corrected_emails():

    print("Loading corrected emails...")

    connection = get_database_connection()

    query = """
        SELECT
            body AS text,
            fixed_emotion AS emotion
        FROM emails
        WHERE fixed_emotion IS NOT NULL
          AND fixed_emotion != ''
    """

    try:
        df = pd.read_sql(query, connection)
    finally:
        connection.close()


    print(df)

    emotion_map = {
        "sadness": 0,
        "joy": 1,
        "disgust": 2,
        "anger": 3,
        "fear": 4
    }

    df["emotion"] = (
        df["emotion"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(emotion_map)
    )

    # Check for values that couldn't be mapped
    if df["emotion"].isna().any():

        invalid_emotions = df.loc[
            df["emotion"].isna(),
            "emotion"
        ]

        print("WARNING: Invalid emotions found:")
        print(invalid_emotions)

        df = df.dropna(subset=["emotion"])

    df["emotion"] = df["emotion"].astype(np.int32)

    print("Corrected dataset:")
    print(df["emotion"].value_counts().sort_index())
    print("dtype:", df["emotion"].dtype)
    print(f"Corrected emails found: {len(df)}")


    return df