import streamlit as st
import os
import cv2
from PIL import Image
import Split_Words
import Split_Characters
import Predict_Characters


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' (default) to display all, '2' to display errors and suppress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

st.title("Digitization of string of middle zone words")
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg","png"])

if uploaded_file is not None:
    Path = 'Words'
    file_name = uploaded_file.name
    # Images = sorted(os.listdir(Path), key = lambda x: int(os.path.splitext(x)[0]))
    # print(Images)
    image = Image.open(uploaded_file)
    # for Image_Name in Images:
    # print(Image_Name)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    # Words = Split_Words.Split(cv2.imread(os.path.join(Path, Image_Name)))
    print('Name of file uploaded: ',file_name)
    Image_cv2= cv2.imread(os.path.join(Path, file_name))
    if Image_cv2 is None:
        print(f"Error: Unable to load image '{file_name}'")
    else:
        Words = Split_Words.Split(Image_cv2)

        Characters = Split_Characters.Split(Words)
        Predictions = Predict_Characters.Predict(Characters)
        Words = []
        for Prediction in Predictions:
            Word = ''.join(Prediction)
            Words.append(Word)
        Words = ' '.join(Words)
            # print(Words)
        st.write("The digitized words are : ",Words)
        with open("devanagari_text.txt", "a", encoding="utf-8") as file:
            # file.write(file_name)
            # file.write("\n")
            file.write(Words)
            file.write("\n")
        print("Done:)")