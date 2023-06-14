import pickle
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

model = load_model('my_model.h5')
X_train = pickle.load(open("X_train.pkl", "rb"))


@st.cache
def predict(nama_produk):
    barang = [nama_produk]
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(barang)
    padded = pad_sequences(sequences, maxlen=21,
                           padding='post', truncating='post')
    result = model.predict(padded)
    if result.argmax() == [0]:
        result = "Dapur"
    elif result.argmax() == [1]:
        result = "Kafe"
    elif result.argmax() == [2]:
        result = "Hobi"
    elif result.argmax() == [3]:
        result = "Elektronik"
    elif result.argmax() == [4]:
        result = "Travel"
    elif result.argmax() == [5]:
        result = "Baju"
    elif result.argmax() == [6]:
        result = "Musik"
    elif result.argmax() == [7]:
        result = "Transportasi"
    elif result.argmax() == [8]:
        result = "Kamera"
    # return render_template("result.html", result=result)
    return jsonify(result)


st.title('Product Segmentation')
st.header('Enter the products name:')
nama_produk = st.text_input()

if st.button("Predict"):
    category = predict(nama_produk)
    st.success(category)
