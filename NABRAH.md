# NABRAH: Happy Customers, Happy Business
<img width="1000" height="500" src="https://github.com/hamadsul/Capstone_Project/blob/main/NABRAH.jpeg">

# Name of the members:-
- Hamad Alsulaiman 
- Norah Almuhaisen
- Sarah Alyami
- Deema Alkhudairi

# Introduction:-
- Our project is utilizing Al to analyze client voice tone for customer satisfaction detection. In today's fast-paced world, it is essential for businesses to keep their customers satisfied. One of the ways to achieve this is by understanding how clients feel about your products or services. This is where Al comes in. By analyzing the tone of a client's voice, Al can help detect customer satisfaction levels and provide valuable insights.

# Dataset Overview:- 
- The Saudi speech detection dataset represents an exceptional resource carefully curated from a wide array of YouTube videos sourced from Telfaz11, a highly renowned Saudi entertainment channel. The primary goal of this meticulous curation process was to capture nuanced impressions, achieved by handpicking scenes that exhibited clear and distinct tonal references. To ensure the highest level of precision, the audio files were meticulously extracted and divided into smaller, contextually relevant segments, ranging from 1 to 9 seconds in length. An important aspect of this dataset is its expert categorization of tones into two distinct types: positive and negative, greatly enhancing its usability and relevance.
- It is crucial to note that the scarcity of Saudi speech datasets further emphasizes the extraordinary value of this particular resource. This scarcity is particularly significant in light of the Saudi government's proactive efforts to enrich Saudi content and promote the development of artificial intelligence applications that specifically support the Arabic language, particularly in the Saudi dialect. Consequently, leveraging this dataset presents a unique opportunity for gaining a competitive advantage in the advancement of speech-related research, development, and innovation within Saudi Arabia and beyond. By utilizing this dataset, developers and researchers can propel the development of cutting-edge speech and audio-related applications in Arabic (Saudi), opening up new horizons for technology-driven solutions in various domains.

# Proposed Algorithms:- 
- This model is a sequential neural network built using the Keras library. It consists of three main layers:
1. LSTM layer with 128 units: LSTM stands for Long Short-Term Memory, which is a type of recurrent neural network (RNN) designed to handle sequence data. This layer is set to return sequences, which means it outputs a sequence of vectors rather than a single vector. The input shape is defined as (X.shape[1:3]), indicating that the input data has two dimensions (sequences and features).
2. LSTM layer with 128 units: This is another LSTM layer with 128 units, but it doesn't return sequences. It takes the output sequences from the previous layer and processes them, providing a condensed representation of the sequence information.
3. Dense layer with 2 units: This is a fully connected layer with 2 units, which means it outputs a vector of size 2. The activation function used in this layer is softmax, which produces a probability distribution over the 2 classes. Softmax ensures that the output values sum up to 1, making it suitable for multi-class classification problems.
- Overall, this model takes sequential data as input and processes it using two LSTM layers, extracting relevant features from the sequence. The final dense layer produces a probability distribution for the input belonging to each of the 6 classes.

# Final Results:-
The final evaluation of the model's performance was carried out using a confusion matrix, yielding the following metrics:
- Accuracy: 0.8590163934426229
- Precision: 0.8670886075949367
- Recall: 0.8616352201257862
- F1 Score: 0.864353312302839
The accuracy metric represents the proportion of correctly classified instances compared to the total number of instances. In this case, the model achieved an accuracy of approximately 85.9%, indicating its overall effectiveness in making correct predictions.

Precision is a measure of the model's ability to correctly identify positive instances out of the total instances predicted as positive. The precision score obtained was approximately 86.7%, indicating a high level of accuracy in identifying true positives.

Recall, also known as sensitivity or true positive rate, measures the model's ability to correctly identify positive instances from the total actual positive instances. The obtained recall score was approximately 86.2%, indicating a satisfactory ability to capture true positives.

The F1 score combines both precision and recall into a single metric, providing a balanced assessment of the model's performance. The F1 score achieved was approximately 86.4%, indicating a favorable trade-off between precision and recall.

These evaluation metrics offer valuable insights into the performance of the model, providing a comprehensive understanding of its accuracy, precision, recall, and overall effectiveness in classification tasks.

# Conclusion:- 
- In conclusion, the use of AI to analyze customer voice tone can provide valuable insights into customer satisfaction levels. The Saudi speech detection dataset is a valuable resource for this purpose, given its expert curation and categorization of tones. The proposed algorithm, a sequential neural network with two LSTM layers and a dense layer, achieved an accuracy of 85.9% in detecting customer satisfaction levels. The precision, recall, and F1 scores also indicate the model's effectiveness in classification tasks. The development of cutting-edge speech and audio-related applications in Arabic (Saudi) can be propelled through the use of this dataset, opening up new horizons for technology-driven solutions in various domains. Overall, this project demonstrates the potential of AI in improving customer satisfaction and enhancing business performance.
