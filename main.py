import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
import streamlit as st
from io import BytesIO
import time
import logging
logging.basicConfig(level=logging.DEBUG)
#load_model = tf.keras.models.load_model('custom_model')
load_model2 = tf.keras.models.load_model('mobilenetv2_model')
uploaded_file = st.file_uploader('Choose a car image', type=["jpg", "png", "jpeg"])
add_selectbox = st.sidebar.selectbox(
"Choose a model to see the car is damaged or not",
("Convolutional Neural Network", "MobilenetV2"))
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.image(uploaded_file)
    np_array = np.frombuffer(file_bytes, np.uint8)
    test_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    test_img = cv2.resize(test_img, (224, 224))
    test_img = tf.expand_dims(test_img, 0)
    col1, col2 = st.columns([1,1])
    col2.button("Reset", type="primary")
    if add_selectbox == 'Convolutional Neural Network':
        if col1.button('Submit'):
            with st.spinner('Wait for it...'):
                time.sleep(1)
                predictions = load_model.predict(test_img)
                score = tf.nn.softmax(predictions[0])
                predicted_class = np.argmax(score)
                relabel_mapping = {
                0: "damage",
                1: "not damage",
                }
                predicted_label = relabel_mapping.get(predicted_class, "Unknown")
                st.markdown(
                    'This image most likely belongs to <span style="color:red; font-weight:bold">{}</span> with a <span style="color:red; font-weight:bold">{:.2f}</span> percent confidence.'
                    .format(predicted_label, 100 * np.max(score)),
                    unsafe_allow_html=True
                )
    elif add_selectbox == "MobilenetV2":
        if col1.button('Submit'):
            with st.spinner('Wait for it...'):
                time.sleep(1)
                predictions = load_model2.predict(test_img)
                score = tf.nn.softmax(predictions[0])
                predicted_class = np.argmax(score)
                relabel_mapping = {
                0: "damage",
                1: "not damage",
                }
                predicted_label = relabel_mapping.get(predicted_class, "Unknown")
                st.markdown(
                    'This image most likely belongs to <span style="color:red; font-weight:bold">{}</span> with a <span style="color:red; font-weight:bold">{:.2f}</span> percent confidence.'
                    .format(predicted_label, 100 * np.max(score)),
                    unsafe_allow_html=True
                )
