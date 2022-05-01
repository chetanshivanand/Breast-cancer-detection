from ctypes import alignment
from tkinter import CENTER
import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

def main():
    primaryColor="black"
    backgroundColor="Blue"
    st.set_page_config(
    page_title="BreastCancerDetector",page_icon= "🎗️")

    st.markdown("<h1 style='text-align: center; color: white ; font-size:65px; font-family: Copperplatec'><b>Breast Cancer Detector</b></h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text_align: center; color: blue;'>This program is designed to predict two severity of abnormalities associated with breast cancer cells: benign and malignant.</h4>", unsafe_allow_html=True)
    st.image("app_pic.png", width=800)


    image_input = st.file_uploader(label="Upload Breast Mammogram (PGM, PNG, JPG)",type=['pgm', 'png', 'jpg'])
    detect = st.button("Detect Breast Cancer")
    np.set_printoptions(suppress=True)

    model = tensorflow.keras.models.load_model('Final_model.h5')
    
    if image_input is not None:
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(image_input, mode='r')
        image = image.convert('RGB')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        data[0] = image_array


        size = st.slider("Adjust Image Size: ", 300, 1000)
        st.image(image, width=size)
        

    if detect:
        prediction = model.predict(data)
        class1 = prediction[0,0]
        class2 = prediction[0,1]
        if class1 > class2:
            st.info(" **Benign Tumor** by {:.2f}%. Please visit [breastcancer.org](https://www.breastcancer.org/) for more information about your next step".format(class1 * 100) )
        elif class2 > class1:
            st.info(" **Malignant Tumor** by {:.2f}%. Please visit [breastcancer.org](https://www.breastcancer.org/) for more information about your next step ".format(class2 * 100))
        else:
            st.info("We encountered an ERROR. This should be temporary, please try again with a better quality image. Cheers!")
    
    st.write("-----------------------------------------------------------")
    st.markdown("<h3 style='text_align: center; color: white; font-size:40px; font-family: Copperplatec'>About Us</h3>",unsafe_allow_html=True)
    st.markdown("<h5 style='text_align: center; color: white; font-family: Copperplatec'><i>This project was made by Dayananda Sagar University students with the supervision of Prof.Ranjini K, to help people detect breast cancer</i</h5>",unsafe_allow_html=True)
    
    
    col1, col2, col3, col4, col5  = st.columns(5)
    with col1:
        st.subheader("***CHETAN S a***")
        
        st.write("check out [LinkedIn](https://www.linkedin.com/in/aya-shehata-0a455b1b6/)")
        st.write("check out [Github](https://github.com/AyaShehata903)")

    with col2:
        st.subheader("***Dharanishree P S***")
        
        st.write("check out [LinkedIn](https://www.linkedin.com/in/omnia-imam)")
        st.write("check out [Github](https://github.com/omnia-emam)")

    with col3:
        st.subheader("***Dhruv Budhiraja***")
       
        st.write("check out [LinkedIn](https://www.linkedin.com/in/kareem-negm)")
        st.write("check out [Github](https://github.com/Kareem-negm)")

    with col4:
        st.subheader("***Chaitanya Srikar***")
        
        st.write("check out [LinkedIn](https://www.linkedin.com/in/kareem-negm)")
        st.write("check out [Github](https://github.com/Kareem-negm)")

    with col5:
        st.subheader("***Hanumesh M***")
        
        st.write("check out [LinkedIn](https://www.linkedin.com/in/kareem-negm)")
        st.write("check out [Github](https://github.com/Kareem-negm)")




if __name__ == '__main__':
    main()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
