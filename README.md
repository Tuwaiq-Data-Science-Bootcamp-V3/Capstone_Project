# Emotion Detection

<p> 
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<!--   <img src="logo.png" width="700" hieght='700' " /> -->
  <img width="8000" height="700" alt="Screen Shot 1444-06-17 at 9 20 35 AM" src="logo.jpg">

</p>




### Phase#2: Dataset 💽
- Pick a suitable dataset that helps you to find reasonable answers to your questions.
- Choose a real dataset that needs to clean and preprocess.
- The dataset should have at least 3000 records for Machine Learning Algorithms / 10,000 for Deep Learning Algorithms.
- Make sure that you really understand your dataset.

### Phase#3: Data Analysis and Preprocessing (Exploratory Data Analysis (EDA)) 🔎📊
- Apply the essential EDA steps: head, shape, info, describe, and missing values.
- Draw at least 10 interactive charts that give an overview of your data.
- The charts should have proper formatting including XY-Axis labels and the main title.



### Phase#5: Model Evaluation (Model Tuning) 🎛
- Report appropriate evaluation metrics for each model.
- Display the used techniques for accuracy enhancement.
- Create a chart that compares the final results of your selected models.


### Phase#7: Conclusion 🏁
- Write a final conclusion and recommendations (your interpretation of the results).


  
# Project Title
Emotion Analysis: A Deep Learning Approach for Speech and Face Emotion Recognition

## Dataset file.
Audio dataset https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## Team Members

- [Name]: [Role]
- [Name]: [Role]
- [Name]: [Role]
- [Name]: [Role]    
    
## Introduction
In this project, we aim to develop a deep learning-based system for emotion recognition from speech and facial expressions. The project consists of two parts: speech emotion recognition and face emotion recognition. For speech emotion recognition, we use the RAVDESS dataset from Kaggle and train a Convolutional LSTM (CLSTM) model to detect emotions in voices. For face emotion recognition, we use the Mediapipe library to extract facial landmarks in real-time from a live camera feed, and use a pre-trained neural network to predict the emotion.  

## Dataset Overview
The RAVDESS dataset consists of speech audio files produced as part of research on speech emotion recognition. The dataset includes 24 actors (12 male, 12 female), and each actor recorded eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, and surprised) in two intensities (normal and strong). The total number of audio files in the dataset is 1,440.
## Proposed Algorithms

### Speech Emotion Recognition

We train a Convolutional LSTM (CLSTM) model to detect emotions in voices from the RAVDESS dataset. The CLSTM model consists of multiple LSTM layers followed by dense layers.

### Face Emotion Recognition

For face emotion recognition, we use the Mediapipe library to extract facial landmarks in real-time from a live camera feed. We use a pre-trained neural network to predict the emotion from the extracted features. The neural network is trained on the FER-2013 dataset, which consists of facial expressions labeled as one of seven categories (angry, disgust, fear, happy, sad, surprise, neutral).

## Final Results and Conclusion

We train and evaluate the CLSTM model for speech emotion recognition and the pre-trained neural network for face emotion recognition. We achieve an accuracy of XX% on the test set for speech emotion recognition and an accuracy of YY% on the test set for face emotion recognition. 

Overall, our project demonstrates the effectiveness of deep learning-based approaches for emotion recognition from speech and facial expressions. Our models can be deployed in real-world applications, such as sentiment analysis in customer service and emotion-based recommendation systems.

