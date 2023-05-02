import cv2 as cv
import streamlit as st
from PIL import Image
import numpy as np

def Median():
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg']) 
    if uploaded_file is not None:
        image=np.array(Image.open(uploaded_file))
    
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
           st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
           st.image(image,width=300)  

        with col2:
           st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True) 
    else:
    
        image = cv.imread("C:/Users/ahmed/Downloads/archive/data/noisyimg.png")
    median = cv.medianBlur(image,5)
    cv.imshow('median',median)
    return median
    cv.waitKey(None)
      
Median()

def edge_filter():
    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg']) 
    if uploaded_file is not None:
       image=np.array(Image.open(uploaded_file))
    
       col1, col2 = st.columns( [0.5, 0.5])
       with col1:
          st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
          st.image(image,width=300)  

       with col2:
          st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True) 
    else:
        image = cv.imread("C:/Users/ahmed/Downloads/archive/data/111.jpg")
    canny = cv.Canny(image,225,275)
    cv.imshow("canny edges" , canny)
    return canny
    cv.waitKey(None)
edge_filter()

def blur_function(image):
    # img = cv.imread("C:/Users/ahmed/Downloads/archive/data/home.jpg")
    blur = cv.GaussianBlur(image,(5,5),0)
    cv.imshow("Blur iamge" , blur)
    return blur
    cv.waitKey(None)  
blur_function()
st.title("DIP Project")
st.markdown("### **This Application allows you to make three diffrent filters to an image**")


# uploaded_file = st.file_uploader("", type=['jpg','png','jpeg']) 
# if uploaded_file is not None:
#     image=np.array(Image.open(uploaded_file))
    
#     col1, col2 = st.columns( [0.5, 0.5])
#     with col1:
#         st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
#         st.image(image,width=300)  

#     with col2:
#         st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True) 
def main():
    filter = st.sidebar.radio('Covert your image to:', ['Blur Filter','Edge Filter', 'Median Filter'])

    if filter=="Blur Filter":
     blur_function()

    if filter=="Edge Filter":
     edge_filter()
    
    if filter=="Median Filter":
     Median()
main()
