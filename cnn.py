
import streamlit as st
import numpy as np
from PIL import Image
from PIL import ImageOps
from keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'


model=load_model('my_model1.h5')
classes=['Basmati', 'Karacadag', 'Arborio', 'Ipsala','Jasmine']

def predict(image):
    img=Image.open(image)
    img=img.resize((70,70))
    img=ImageOps.grayscale(img)
    img_array = np.array(img)
    img_array = img_array.reshape((1,70, 70, 1))
    prediction=model.predict(img_array)
    predicted_class=np.argmax(prediction)
    return classes[predicted_class]

st.title("RICE CLASSIFICATION")

uploaded_file=st.file_uploader("Choose an image.......",type=['png','jpg','jpeg'])
if uploaded_file is not None:
    st.write("Classifying.......:point_down:")
    class_name = predict(uploaded_file)
    st.markdown(f'<span style="color:blue; font-size:30px;"><b>{class_name}</b></span>', unsafe_allow_html=True)
    if class_name == "Basmati":
        st.write("Basmati")
    elif class_name == "Karacadag":
        st.write('Karacadag')
    elif class_name == "Arborio":
        st.write('Arborio')
    elif class_name == "Ipsala":
        st.write('Ipsala')
    elif class_name == "Jasmine":
        st.write('Jasmine')
