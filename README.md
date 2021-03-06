# Group 1 RecognizeMe App

### For this project, our team built a Facial Recognition application that is able to detect our teammates' faces and welcome us into our respective "accounts." 

The application consists of three supervised convolutional neural networks:
#### 1) Human Face Detection Model 
This model was trained on a set of human and non-human images. It is able to recognize whether an object in the webcam is a human, and, if not, return a message stating that it only works on human faces.
#### 2) Teammates' Face Detection Model
We trained this model using a dataset with images of our own faces with corresponding labels, as well the Flickr Faces HQ dataset of human face images. This allowed the application to identify if any of the teammembers are attempting to log in and log us into our respective accounts. 
#### 3) Spoof Detection Model
This final model allows the application to detect whether it is being fooled by a photo being held up to the camera or if it is truly the user logging in. 

### Please refer to the below links to find the datasets used to train these models:

Human & Non-human Images (used for Model #1) https://www.kaggle.com/aliasgartaksali/human-and-non-human

Flickr-Faces-HQ Dataset (FFHQ) (used for Model #2) - https://drive.google.com/drive/folders/1QNOG0eSWZH9u2310Z9I7AhOy6M_UVP5K
