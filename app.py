import cv2
import streamlit as st 
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob
style_lis = glob.glob('styles/*.jpg')
images = [Image.open(x) for x in style_lis]


@st.cache()
def load_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return model
model = load_model()


st.title("Style Cam")
st.header('Neural Style Transfer')
st.sidebar.title('Parameters')
menu=st.sidebar.selectbox("Menu",['Styles','Custom','Custom Cam'])

if menu == 'Custom Cam':
    st.subheader('Choose Style Image')
    
    file_up = st.empty()
    dp = st.empty()


    img = file_up.file_uploader('Style Image',type=['jpg','jpeg'])
    
    if img:
        img = Image.open(img)
        dp.image(img,use_column_width=True,caption="Style Image")
        style_image = np.asarray(np.array(img)).astype('float32')/255
        style_image = style_image[tf.newaxis, :]

        run = st.checkbox('Run')
        camera = cv2.VideoCapture(0)
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = np.asarray(np.array(frame)).astype('float32')/255
            input_image = input_image[tf.newaxis, :]

            res = np.squeeze(model(tf.constant(input_image), tf.constant(style_image))[0])
            dp.image(res,use_column_width=True,width=250,height=250)

    else:
        st.write('Stopped')





elif menu =='Styles':

    st.sidebar.header('Choose the style')
    style = st.sidebar.selectbox('Style',[x.split('\\')[-1] for x in style_lis])
    for im,name in zip(images,[x.split('\\')[-1] for x in style_lis]):
        st.sidebar.image(im,use_column_width=True ,caption=name)
    style_image = np.asarray(np.array(Image.open(f"styles/{style}"))).astype('float32')/255
    style_image = style_image[tf.newaxis, :]
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = np.asarray(np.array(frame)).astype('float32')/255
        input_image = input_image[tf.newaxis, :]

        res = np.squeeze(model(tf.constant(input_image), tf.constant(style_image))[0])
        FRAME_WINDOW.image(res,use_column_width=True,width=250,height=250)

    else:
        st.write('Stopped')
    
elif menu == 'Custom':

    st.subheader('Choose Style Image')
    
    transfer = st.button('Style Stranfer')
    file_up = st.empty()
    dp = st.empty()


    img = file_up.file_uploader('Style Image',type=['jpg','jpeg'])
    
    if img:
        img = Image.open(img)
        dp.image(img,use_column_width=True,caption="Style Image")

        img2 = file_up.file_uploader('Input Image',type='jpg')
        
        if img2:
            img2 = Image.open(img2)
            dp.image(img2,use_column_width=True,caption="Style Image")

            input_image = np.asarray(np.array(img2)).astype('float32')/255
            input_image = input_image[tf.newaxis, :]

            style_image = np.asarray(np.array(img)).astype('float32')/255
            style_image = style_image[tf.newaxis, :]

            if transfer:
                res = np.squeeze(model(tf.constant(input_image), tf.constant(style_image))[0])

                dp.image(res,use_column_width=True,width=250,height=250)
            
