import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

process_dir = os.getcwd()

def load_image(image_file):
    img = Image.open(image_file)
    return img 


def chest_sickness_predict(image):
    if "png" in str(image):
        im = Image.open(image)
        rgb_im = im.convert('RGB')
        img2 = os.path.join(process_dir, "test.jpeg")
        rgb_im.save(img2)

    else:
        img1 = Image.open(image)
        img2 = os.path.join(process_dir, "test.jpeg")
        img1.save(img2)

    img = cv2.imread(img2)
    chest_sickness_model = load_model('medical_trial_model.h5')
    print('X-Ray Model->', chest_sickness_model)
    test_image = cv2.resize(img,(224,224))
    test_image = np.array(test_image)
    test_image = np.expand_dims(test_image, axis= 0)
    test_image = test_image/255
    result = chest_sickness_model.predict(test_image)

    if result[0][0] > 0.8:
        print(result[0][0])
        print(result)
        print(result[0])
        prediction = "COVID"
        confidence_score = result[0][0] * 100
        print(f'confidence score de covid -> {confidence_score}')
    elif result[0][1] > 0.8:
        print(result[0][1])
        print(result)
        print(result[0])
        prediction = "NORMAL"
        confidence_score = result[0][1] * 100
        print(f'confidence score de normal -> {confidence_score}')
    elif result[0][2] > 0.8:
        print(result[0][0])
        prediction = "VIRUS RESPIRATORIO"
        confidence_score = result[0][2] * 100
        print(f'confidence score de virus respiratorio -> {confidence_score}')

    return prediction, confidence_score


def main():
    st.title('**CT X-Rays Predictions**')
    html_temp = """
    <div style="background-color:red;padding;10px; text-align:center">
    <h2>Predicción de Pulmon</h2>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    filename = st.file_uploader("SUBE LA IMAGEN CT", type=['jpg', 'png','jpeg','gif'])

    if ((filename is not None) and ('jpg' in str(filename))):
        st.image(load_image(filename))
        prediction, confidence_score = chest_sickness_predict(filename)
        confidence_score = str(confidence_score)
        

    elif ((filename is not None) and ('png' in str(filename))):
        st.image(load_image(filename))
        prediction, confidence_score = chest_sickness_predict(filename)
        confidence_score = str(confidence_score)
    

    elif ((filename is not None) and ('jpeg' in str(filename))):
        st.image(load_image(filename))
        prediction, confidence_score = chest_sickness_predict(filename)
        confidence_score = str(confidence_score)

    elif ((filename is not None) and ('gif' in str(filename))):
        st.image(load_image(filename))
        prediction, confidence_score = chest_sickness_predict(filename)
        confidence_score = str(confidence_score)

    else:
        pass

    result=""
    if st.button("Predicción"):
       #st.success('Prediction: {} , Confidence Score(%): {}',format(prediction, confidence_score))
       st.success(f'Predicción:  {prediction}')
       st.success(f'Probabilidad del Arrojada por el Modelo:  {confidence_score} %')
    



if __name__ == '__main__':
    main()
