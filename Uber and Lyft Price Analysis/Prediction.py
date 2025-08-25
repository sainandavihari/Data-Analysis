import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
import spacy
import streamlit as st

def get_dict_map(data, token_or_tag):
    vocab = list(set(data[token_or_tag].to_list()))
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

def get_bilstm_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim))
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    st.title("Named Entity Recognition System")
    text_input = st.text_area("Enter text for NER analysis")
    if st.button("Analyze"):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text_input)
        # Process text and display results
        st.write(displacy.render(doc, style='ent'))

if __name__ == '__main__':
    main()


