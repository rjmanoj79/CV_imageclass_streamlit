# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:14:11 2020

@author: abc
"""
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from img_classification import teachable_machine_classification

st.header("Brain Tumor MRI Classification")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    #label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.write("The MRI scan has a brain tumor")
    else:
        st.write("The MRI scan is healthy")
