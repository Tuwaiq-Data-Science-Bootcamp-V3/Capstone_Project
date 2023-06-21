# This is a sample Python script.
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cvzone
import base64
from streamlit_lottie import st_lottie
from PIL import Image
import requests
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from ultralytics import YOLO
import cv2
import numpy as np

class VideoTransformer(VideoTransformerBase):

    def transform(self, frame):
        model = YOLO(r'C:\Users\fai-w\Desktop\weights\best.pt')

        img = frame.to_ndarray(format='bgr24')
        #cam = cv2.VideoCapture(0)
        # width
        #cam.set(3, 1280)
        # height
        #cam.set(4, 720)

        # classes
        names = ['Hoodie', 'Jacket', 'Mid-length dress', 'Pants', 'Tshirt', 'dress', 'skirt']
        # LOWER AND UPPER RANGE OF MULTIPLE COLORS

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([250, 255, 30])

        lower_white = np.array([0, 0, 255])
        upper_white = np.array([0, 0, 255])

        lower_red = np.array([0, 150, 50])
        upper_red = np.array([10, 255, 255])

        #lower_green = np.array([45, 150, 50])
        #upper_green = np.array([65, 255, 255])
        lower_green = np.array([25, 52, 72])
        upper_green = np.array([102, 255, 255])

        lower_yellow = np.array([25, 150, 50])
        upper_yellow = np.array([35, 255, 255])

        lower_light_blue = np.array([95, 150, 0])
        upper_light_blue = np.array([110, 255, 255])

        lower_orange = np.array([15, 150, 0])
        upper_orange = np.array([25, 255, 255])


       # lower_dark_pink = np.array([160, 150, 0])
       # upper_dark_pink = np.array([170, 255, 255])

        #lower_pink = np.array([145, 150, 0])
        #upper_pink = np.array([155, 255, 255])
        lower_pink = np.array([181,126,220])
        upper_pink = np.array([220,208,255])

        lower_cyan = np.array([85, 150, 0])
        upper_cyan = np.array([95, 255, 255])

        lower_dark_blue = np.array([115, 150, 0])
        upper_dark_blue = np.array([125, 255, 255])

        # purple
        lower_purple = np.array([129, 50, 70])
        # np.array([129, 50, 70])138,43,226
        upper_purple = np.array([158, 255, 255])
        # np.array([158, 255, 255])255,0,255
        # light purple
        # upper_light_purple = np.array([220,208,255])
        # lower_light_purple = np.array([181,126,220])
        # dark purple
        # upper_dark_purple = np.array([145,92,131])
        # lower_dark_purple = np.array([50,23,77])

        # turqise
        lower_turq = np.array([0, 206, 209])
        upper_turq = np.array([175, 238, 238])

        # TO KEEP THE WEBCAM RUNNING
        while True:
            # IF CAM IS FOUND, SAVE IT TO "img"
            #success, img = cam.read()

            # YOLO MODEL RESULTS
            results = model(img, stream=True)
            # CAPTURE COLORS AND TURN THEM TO "HSV" VALUES
            imgC = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # range of colors
            yellow = cv2.inRange(imgC, lower_yellow, upper_yellow)
            red = cv2.inRange(imgC, lower_red, upper_red)
            dblue = cv2.inRange(imgC, lower_dark_blue, upper_dark_blue)
            green = cv2.inRange(imgC, lower_green, upper_green)
            pink = cv2.inRange(imgC, lower_pink, upper_pink)
            lblue = cv2.inRange(imgC, lower_light_blue, upper_light_blue)
            purple = cv2.inRange(imgC, lower_purple, upper_purple)
            orange = cv2.inRange(imgC, lower_orange, upper_orange)
            cyan = cv2.inRange(imgC, lower_cyan, upper_cyan)
            turq = cv2.inRange(imgC, lower_turq, upper_turq)

            # lpurple = cv2.inRange(imgC, lower_light_purple, upper_light_purple)
            # dpurple = cv2.inRange(imgC, lower_dark_purple, upper_dark_purple)
            # morphological transformation and dilation
            # kernal=np.ones((5,5), 'uint8')
            # red=cv2.dilate(red, kernal)
            # resr=cv2.bitwise_and(img, img, mask = red)
            # blue=cv2.dilate(blue, kernal)
            # resb=cv2.bitwise_and(img, img, mask = blue)
            # yellow=cv2.dilate(yellow, kernal)
            # resy=cv2.bitwise_and(img, img, mask = red)

            # tracking colors
            (contours, hierarchy) = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,205), 2)
                    cv2.putText(img, 'red', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,205), 2)

            # BLUE
            (contours, hierarchy) = cv2.findContours(dblue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,205), 2)
                    cv2.putText(img, 'dark blue', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,205), 2)
            # YELLOW
            (contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 35), 2)
                    cv2.putText(img, 'yellow', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 35), 2)

            # green
            (contours, hierarchy) = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (25, 255, 10), 2)
                    cv2.putText(img, 'green', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 255, 10), 2)
            # pink
            (contours, hierarchy) = cv2.findContours(pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255,105,180), 2)
                    cv2.putText(img, 'pink', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,105,180), 2)
            # lblue
            (contours, hierarchy) = cv2.findContours(lblue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 110), 2)
                    cv2.putText(img, 'light blue', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 110), 2)

            # purple
            (contours, hierarchy) = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (200,105,255), 2)
                    cv2.putText(img, 'purple', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,105,255), 2)
            # ORANGE
            (contours, hierarchy) = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (1,180,255), 2)
                    cv2.putText(img, 'orange', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1,180,255), 2)
            # CYAN
            (contours, hierarchy) = cv2.findContours(cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 95), 2)
                    cv2.putText(img, 'cyan', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 95), 2)

            # TURQ
            (contours, hierarchy) = cv2.findContours(turq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pic, conn in enumerate(contours):
                area = cv2.contourArea(conn)
                if (area > 500):
                    x, y, w, h = cv2.boundingRect(conn)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (64,224,208), 2)
                    cv2.putText(img, 'turq', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64,224,208), 2)
            # cv2.imshow('mask', mask)
            # cv2.imshow('mask', mask)

            # cv2.imshow('mask', mask)
            # contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # if len(contour)!=0:
            #    for cont in contour:
            #        if cv2.contourArea(cont)>1000:
            #            x, y, w, h=cv2.boundingRect(cont)
            #            cv2.rectangle(img, (x, y), (x+w,y+h), (251, 255, 255), 3)
            #            cv2.putText(img, 'yellow', (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (251,255,255), 2)
            # cam.set(CV_CAP_PROP_BUFFERSIZE, 3)
            # results.boxes
            # IF CAN NOT THERE
            # if not cam.isOpened():

            #     print('Unable to load camera.')
            #     sleep(5)
            #     pass
            # FOR CAPTURING BOUNDRIES OF THE YOLO PREDICTIONS
            # Capture frame-by-frame
            # ret, frame = cam.read()
            # end of extra
            # loop through every frame
            for r in results:
                boxes = r.boxes
                # get Xs and Ys of each box
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w1, h1 = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, x2, y2))
                    # define confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # define class
                    cls = int(box.cls[0])
                    # show confidence and classes
                    cv2.putText(img, f'{names[cls]} {conf}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    #cvzone.putTextRect(img, f'{names[cls]} {conf}', (max(0, x1), max(35, y1)))
                # print(x1, y1, x2, y2)
            #cv2.imshow("image", img)
        # TO QUIT WEBCAM, PRESS "q" (SOMETIMES YOU HAVE TO HOLD IT AND OTHER TIMES YOU CAN JUST PRESS IT)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    # cam.release()
    # cv2.destroyAllWindows()
            return img
#import streamlit as st
def main():
    st.set_page_config(page_title='ColorCorrectness', layout='wide')

    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)

    mystyle = '''
        <style>
            p {
                text-align: right;
            }
        </style>
        '''

    st.markdown(mystyle, unsafe_allow_html=True)
    mystyleh2 = '''
            <style>
                h2 {
                    text-align: right;
                }
            </style>
            '''

    st.markdown(mystyleh2, unsafe_allow_html=True)
    mystyleh3 = '''
                <style>
                    h3 {
                        text-align: right;
                    }
                </style>
                '''

    st.markdown(mystyleh3, unsafe_allow_html=True)
    set_background('D:/Blue_modern_Company_Zoom_Virtual_Background.png')

   # def load_lottieurl(url):
   #     r = requests.get(url)
   #     if r.status_code != 200:
   #         return None
   #     return r.json()

    # Use local CSS
   # def local_css(file_name):
   #     with open(file_name) as f:
   #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    #local_css(r"style/style.css")

    # ---- LOAD ASSETS ----
    #lottie_coding = load_lottieurl(r"https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
    #img_contact_form = Image.open(r"images/yt_contact_form.png")
    #img_lottie_animation = Image.open(r"images/yt_lottie_animation.png")

    # ---- HEADER SECTION ----
    with st.container():
        st.markdown("<h1 style='text-align: center; color: white;'>ColorCorrectness | مصحح الألوان</h1>", unsafe_allow_html=True)
        #st.title("ColorCorrectness | مصحح الألوان")
        st.write("")
        st.write("")
        st.markdown("<h3 style='text-align: center; color: white;'>!مصحح الألوان هو مساعدك الشخصي لحل تساؤلات جميع من يعانون من عمى الألوان حول ألوان الملابس التي يرتدونها</h1>", unsafe_allow_html=True)
        #st.subheader("!مصحح الألوان هو مساعدك الشخصي لحل تساؤلات جميع من يعانون من عمى الألوان حول ألوان الملابس التي يرتدونها")
        #st.write("Color Correctness is a personal helper for everyone with color blindness where it defines colors for the piece of clothes you are wearing!")


    # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with right_column:
            #st.header("Color Correctness Guide:")
            st.header("إرشادات مصحح الألوان")
            st.write("##")
            #st.write(
            #    """
            #    What you should to do:
            #    - Press "start" to open your webcam.
            #    - Put the piece of clothes you want to know the colors of.
            #    - Done! the model will highlight your piece of clothes and name it's color.
        

            #    Give it a shot!
            #    """
            #)
            st.subheader(
                """
                :ما يجب عليك فعله                                        
                """
            )
            st.subheader("""
                  . لتفعيل الكاميرا  "START" اضغط على زر  -                       
                
            """)
            st.subheader("""
                              .اظهر لباسك للكاميرا حتى تتعرف الكاميرا على لون الملبس  -

                        """)
            st.subheader("!قم بتجربتها")

        with left_column:
            video1 = open(r"D:\Untitled_video_-_Made_with_Clipchamp.mp4", "rb")
            st.video(video1)
            #st_lottie(lottie_coding, height=300, key="coding")
    # ---- PROJECTS ----
    with st.container():
        st.write("---")
        st.header("Color Correctness Model | مودل مصحح الألوان ")
        st.write("##")
        webrtc_streamer(key='example', video_transformer_factory=VideoTransformer)

    with st.container():
        st.write("---")
        st.header("Meet Our Team! | !تعرف على فريقنا")
        st.write("##")
        col1, col2, col3, col4 = st.columns(4)
    #with st.container():

        with col1:
            #st.header("A cat")
            image1 = Image.open(r"D:\Albandari_Card1.jpg")

            st.image(image1)
        with col2:
            #st.header("A dog")
            #image1 = open(r"D:\Najlaa_Card_.jpg", "rb")
            #st.image(image1)
            image2 = Image.open(r"D:\Amjad_Card1.jpg")

            st.image(image2)
            #st.image(r"D:\Najlaa_Card_.jpg")

        with col3:
            #st.header("An owl")
            image3 = Image.open(r"D:\Albandari_Card1.jpg")

            st.image(image3)
            #st.image("D:\Albandari_Card_.png")
        with col4:
            #st.header("A cat")
            image4 = Image.open(r"D:\Fai_Card1_.jpg")

            st.image(image4)
            #st.image(r"D:\Najlaa_Card1.jpg")
    #
    #       st.write('---')
    #        left_column, right_column = st.columns(2)
    #        with left_column:
    #            st.header()

        #image_column, text_column = st.columns((1, 2))
        #with image_column:
        #    st.image(img_lottie_animation)
        #with text_column:
        #    st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
        #    st.write(
        #        """
        #        Learn how to use Lottie Files in Streamlit!
        #        Animations make our web app more engaging and fun, and Lottie Files are the easiest way to do it!
        #        In this tutorial, I'll show you exactly how to do it
         #       """
         #   )
         #   st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")
    #with st.container():
    #    image_column, text_column = st.columns((1, 2))
    #    #with image_column:
        #    st.image(img_contact_form)
    #    with text_column:
    #        st.subheader("How To Add A Contact Form To Your Streamlit App")
    #        st.write(
    #            """
    #            Want to add a contact form to your Streamlit website?
    #            In this video, I'm going to show you how to implement a contact form in your Streamlit app using the free service ‘Form Submit’.
    #            """
    #        )
    #        st.markdown("[Watch Video...](https://youtu.be/FOULV9Xij_8)")

    # ---- CONTACT ----

#with st.container():
#        st.write("---")
#        st.header("Get In Touch With Me!")
#        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
#        contact_form = """
#        <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
#            <input type="hidden" name="_captcha" value="false">
#            <input type="text" name="name" placeholder="Your name" required>
 #           <input type="email" name="email" placeholder="Your email" required>
 #           <textarea name="message" placeholder="Your message here" required></textarea>
 #           <button type="submit">Send</button>
 #       </form>
 #       """
 #       left_column, right_column = st.columns(2)
 #       with left_column:
 #           st.markdown(contact_form, unsafe_allow_html=True)
 #       with right_column:
 #           st.empty()

    #with st.container():
    #    st.title("Color Correctness  |  مصحح الألوان")
       # st.write("Color Correctness is a personal helper for everyone with color blindness where it defines colors for the piece of clothes you are wearing!")

#    with st.container():
#        st.write('---')
#        left_column, right_column = st.columns(2)
#        with left_column:
#            st.header()
#        webrtc_streamer(key='example', video_transformer_factory=VideoTransformer)



#    model=YOLO(r'C:\Users\fai-w\Desktop\weights\best.pt')

#    cam = cv2.VideoCapture(0)
    # width
#    cam.set(3, 1280)
    # height
#    cam.set(4, 720)

    #classes
#    names=['Hoodie', 'Jacket', 'Mid-length dress', 'Pants', 'Tshirt', 'dress','skirt']
    # LOWER AND UPPER RANGE OF MULTIPLE COLORS

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([250, 255, 30])

    lower_white = np.array([0, 0, 255])
    upper_white = np.array([0, 0, 255])

    lower_red = np.array([0, 150, 50])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([45, 150, 50])
    upper_green = np.array([65, 255, 255])

    lower_yellow = np.array([25, 150, 50])
    upper_yellow = np.array([35, 255, 255])

    lower_light_blue = np.array([95, 150, 0])
    upper_light_blue = np.array([110, 255, 255])

    lower_orange = np.array([15, 150, 0])
    upper_orange = np.array([25, 255, 255])

    lower_dark_pink = np.array([160, 150, 0])
    upper_dark_pink = np.array([170, 255, 255])

    lower_pink = np.array([145, 150, 0])
    upper_pink = np.array([155, 255, 255])

    lower_cyan = np.array([85, 150, 0])
    upper_cyan = np.array([95, 255, 255])

    lower_dark_blue = np.array([115, 150, 0])
    upper_dark_blue = np.array([125, 255, 255])

# purple
    lower_purple = np.array([129, 50, 70])
# np.array([129, 50, 70])138,43,226
    upper_purple = np.array([158, 255, 255])
# np.array([158, 255, 255])255,0,255
# light purple
# upper_light_purple = np.array([220,208,255])
# lower_light_purple = np.array([181,126,220])
# dark purple
# upper_dark_purple = np.array([145,92,131])
# lower_dark_purple = np.array([50,23,77])


# turqise
    lower_turq = np.array([0, 206, 209])
    upper_turq = np.array([175, 238, 238])

# TO KEEP THE WEBCAM RUNNING
#    while True:
    #IF CAM IS FOUND, SAVE IT TO "img"
#        success, img = cam.read()

    #YOLO MODEL RESULTS
#        results = model(img, stream=True)
    # CAPTURE COLORS AND TURN THEM TO "HSV" VALUES
#        imgC = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#        #range of colors
#        yellow = cv2.inRange(imgC, lower_yellow, upper_yellow)
#        red = cv2.inRange(imgC, lower_red, upper_red)
#        dblue = cv2.inRange(imgC, lower_dark_blue, upper_dark_blue)
#        green = cv2.inRange(imgC, lower_green, upper_green)
#        pink = cv2.inRange(imgC, lower_pink, upper_pink)
#        lblue = cv2.inRange(imgC, lower_light_blue, upper_light_blue)
#        purple = cv2.inRange(imgC, lower_purple, upper_purple)
#        orange = cv2.inRange(imgC, lower_orange, upper_orange)
#        cyan = cv2.inRange(imgC, lower_cyan, upper_cyan)
#        turq = cv2.inRange(imgC, lower_turq, upper_turq)

    # lpurple = cv2.inRange(imgC, lower_light_purple, upper_light_purple)
    # dpurple = cv2.inRange(imgC, lower_dark_purple, upper_dark_purple)
    # morphological transformation and dilation
    # kernal=np.ones((5,5), 'uint8')
    # red=cv2.dilate(red, kernal)
    # resr=cv2.bitwise_and(img, img, mask = red)
    # blue=cv2.dilate(blue, kernal)
    # resb=cv2.bitwise_and(img, img, mask = blue)
    # yellow=cv2.dilate(yellow, kernal)
    # resy=cv2.bitwise_and(img, img, mask = red)

    # tracking colors
#        (contours, hierarchy) = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#           area = cv2.contourArea(conn)
#           if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'red', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)

    # BLUE
#        (contours, hierarchy) = cv2.findContours(dblue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'dark blue', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)
    # YELLOW
#        (contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'yellow', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)

    # green
#        (contours, hierarchy) = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'green', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)
    # pink
#        (contours, hierarchy) = cv2.findContours(pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'pink', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)
#    # lblue
#        (contours, hierarchy) = cv2.findContours(lblue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'light blue', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)

    # purple
#        (contours, hierarchy) = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'purple', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)
    # ORANGE
#        (contours, hierarchy) = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'orange', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)
    # CYAN
#        (contours, hierarchy) = cv2.findContours(cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#            area = cv2.contourArea(conn)
#            if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'cyan', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)

    # TURQ
#        (contours, hierarchy) = cv2.findContours(turq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        for pic, conn in enumerate(contours):
#           area = cv2.contourArea(conn)
#           if (area > 500):
#                x, y, w, h = cv2.boundingRect(conn)
#                cv2.rectangle(img, (x, y), (x + w, y + h), (251, 255, 255), 2)
#                cv2.putText(img, 'turq', (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (251, 255, 255), 2)
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask', mask)

    # cv2.imshow('mask', mask)
    # contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contour)!=0:
    #    for cont in contour:
    #        if cv2.contourArea(cont)>1000:
    #            x, y, w, h=cv2.boundingRect(cont)
    #            cv2.rectangle(img, (x, y), (x+w,y+h), (251, 255, 255), 3)
    #            cv2.putText(img, 'yellow', (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (251,255,255), 2)
    # cam.set(CV_CAP_PROP_BUFFERSIZE, 3)
    # results.boxes
    # IF CAN NOT THERE
   # if not cam.isOpened():

   #     print('Unable to load camera.')
   #     sleep(5)
   #     pass
    # FOR CAPTURING BOUNDRIES OF THE YOLO PREDICTIONS
    # Capture frame-by-frame
    #ret, frame = cam.read()
    # end of extra
    # loop through every frame
#        for r in results:
#            boxes = r.boxes
        # get Xs and Ys of each box
#            for box in boxes:
#                x1, y1, x2, y2 = box.xyxy[0]
#                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                w1, h1 = x2-x1, y2-y1
#                cvzone.cornerRect(img, (x1, y1, x2, y2))
            #define confidence
#                conf = math.ceil((box.conf[0]*100))/100
            #define class
#                cls = int(box.cls[0])
            #show confidence and classes
#                cvzone.putTextRect(img, f'{names[cls]} {conf}', (max(0, x1), max(35, y1)))
            #print(x1, y1, x2, y2)
#        cv2.imshow("image", img)
    # TO QUIT WEBCAM, PRESS "q" (SOMETIMES YOU HAVE TO HOLD IT AND OTHER TIMES YOU CAN JUST PRESS IT)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#cam.release()
#cv2.destroyAllWindows()
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
