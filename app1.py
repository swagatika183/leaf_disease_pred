import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO,StringIO
import requests
from PIL import Image,ImageOps
import pickle as pkl





def main():
    #st.title("AI Demo")
    #st.subheader("High Level Analysis")
    html_temp = """ 
    <div style="background.color:green;padding:15px;">
    <h2>Bean Image Classifier</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
      
            
        


main()
    

MODEL_PATH="model_mob.h5"
from keras.models import load_model
import tensorflow_hub as hub
model = load_model(MODEL_PATH,custom_objects={'KerasLayer':hub.KerasLayer})
@st.cache(allow_output_mutation=True)

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH,custom_objects={'KerasLayer':hub.KerasLayer})
    return model





classes = ["angular_leaf_spot","bean_rust","Healthy"]

def scale(image):
    image = tf.cast(image,tf.float32)
    image /=255.0

    return tf.image.resize(image,[224,224])

def decode_img(image):
    img = tf.image.decode_jpeg(image,channels=3)
    img = scale(img)
    return np.expand_dims(img,axis=0)

#path = st.text_input("Enter Image URL to classify..")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

    
filename = st.file_uploader("Please Upload file",type=["csv","png","jpg"])
                     
def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape =img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    

    #with st.spinner("classifying....."):
        #label = np.argmax(model.predict(decode_img(image_data)),axis=1)
    return prediction
    

#if file is None:
   # st.text("please upload an image file")
#else:
#filename = file_selector()
st.write('You selected `%s`' % filename)
image = Image.open(filename)
st.image(image,use_column_width=True)

#label = np.argmax(model.predict(decode_img(image)),axis=1)
#st.write(classes[label[0]])
#st.write("")
#img = Image.open(BytesIO(image))
#st.image(img,caption="Classifying Bean Image",use_column_width=True)
prediction =import_and_predict(image,model)
string ="This image most likely is: " +classes[np.argmax(prediction)]
st.success(string)
    
     








 
