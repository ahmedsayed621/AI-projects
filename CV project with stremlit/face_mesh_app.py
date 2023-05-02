import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.mp4'



st.title('DIP Detection App')

st.markdown (
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
      width:350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
      width:350px
      margin-left:-350px
    }
    </style>
    
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('DIP SideBar')
st.sidebar.subheader('parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))
    #resize the image

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox('Choose the app mode:',
                                ['About app', 'Face Detection','thresholding' ,'Edge Detection'
                                ,'Median And gussien filter','sharpening filter'])

if app_mode =='About app':
   st.markdown(' this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
   

elif app_mode == 'Face Detection':
    drawing_spec=mp_drawing.DrawingSpec(thickness=2,circle_radius=1)
    st.sidebar.markdown('------')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
          width:350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
          width:350px
          margin-left:-350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
    st.subheader("Face Detection")
    st.markdown("**Detected faces**")
    kpi1_text = st.markdown("0")
    max_faces= st.sidebar.number_input("Max number of faces",value=2,min_value=1)
    st.sidebar.markdown("-----")
    detection_confidence=st.sidebar.slider("Min Detection Confidence" , min_value=0.0,max_value=1.0 , value=0.5)
    st.sidebar.markdown("-----")
    img_file_buffer = st.sidebar.file_uploader("Upload an image ",type=["jpg","png","jpeg"])
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))
    else:
        demo_image=DEMO_IMAGE
        image=np.array(Image.open(demo_image))

    st.sidebar.text("Original Image")
    st.sidebar.image(image)

    face_count = 0


    ##Dashboard

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        min_detection_confidence= detection_confidence) as face_mesh:

            result = face_mesh.process(image)
            out_image = image.copy()

            ##for landmark drawing
            for face_landmarks in result.multi_face_landmarks:
                face_count +=1

                mp_drawing.draw_landmarks(
                    image=out_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec
                )
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>",unsafe_allow_html=True)
            st.subheader('output image')
            st.image(out_image,use_column_width=True)


elif app_mode == 'thresholding':
    st.sidebar.markdown('------')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
          width:350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
          width:350px
          margin-left:-350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
    
    img_file_buffer = st.sidebar.file_uploader("Upload an image ",type=["jpg","png","jpeg"])
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))
    else:
        demo_image=DEMO_IMAGE
        image=np.array(Image.open(demo_image))

    st.sidebar.text("Original Image")
    st.sidebar.image(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    x = st.slider('Change Threshold value',min_value = 50,max_value = 255)  

    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
    thresh1 = thresh1.astype(np.float64)
    st.image(thresh1, use_column_width=True,clamp = True)
    
    st.text("Bar Chart of the image")
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    st.bar_chart(histr)

    
elif app_mode == 'Edge Detection':
    st.sidebar.markdown('------')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
          width:350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
          width:350px
          margin-left:-350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
    
    img_file_buffer = st.sidebar.file_uploader("Upload an image ",type=["jpg","png","jpeg"])
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))
    else:
        demo_image=DEMO_IMAGE
        image=np.array(Image.open(demo_image))

    st.sidebar.text("Original Image")
    st.sidebar.image(image)

    st.subheader(" view Canny Edge Detection Technique")
    edges = cv2.Canny(image,50,300)
    cv2.imwrite('edges.jpg',edges)
    st.image(edges,use_column_width=True,clamp=True)



elif app_mode == 'Median And gussien filter':
    st.sidebar.markdown('------')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
          width:350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
          width:350px
          margin-left:-350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
    
    img_file_buffer = st.sidebar.file_uploader("Upload an image ",type=["jpg","png","jpeg"])
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))
    else:
        demo_image=DEMO_IMAGE
        image=np.array(Image.open(demo_image))

    st.sidebar.text("Original Image")
    st.sidebar.image(image)

    st.subheader(" view Filters Technique")
    median = cv2.medianBlur(image, 5)
    gauss = cv2.GaussianBlur(image, (5,5), 0)

    images = np.concatenate((median, gauss), axis=1)

    st.image(images)
    st.text("Left: Median filtering. Right: Gaussian filtering.")



elif app_mode == 'sharpening filter':
    st.sidebar.markdown('------')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
          width:350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
          width:350px
          margin-left:-350px
        }
        </style>

        """,
        unsafe_allow_html=True,
    )
    
    img_file_buffer = st.sidebar.file_uploader("Upload an image ",type=["jpg","png","jpeg"])
    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))
    else:
        demo_image=DEMO_IMAGE
        image=np.array(Image.open(demo_image))

    st.sidebar.text("Original Image")
    st.sidebar.image(image)


    st.subheader(" view sharpening Filter Technique")
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    st.image(image_sharp)


    # # Obtain number of rows and columns
    # # of the image
    # img=cv2.imread(image,0)
    # m, n = img.shape
  
    # # Develop Averaging filter(3, 3) mask
    # mask = np.ones([3, 3], dtype = int)
    # mask = mask / 9
  
    # # Convolve the 3X3 mask over the image
    # img_new = np.zeros([m, n])
 
    # for i in range(1, m-1):
    #   for j in range(1, n-1):
    #     temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
        
    #     img_new[i, j]= temp
         
    # img_new = img_new.astype(np.uint8)
    # st.image(img_new)



