# SMS Spam Classification

This project is focused on building a machine learning model to classify SMS messages as either "spam" or "ham" (not spam). The model helps in detecting and filtering out unwanted spam messages, ensuring a cleaner and more relevant inbox for users.

## Project Overview
The primary objective of this project is to develop a robust classifier that can accurately distinguish between spam and ham messages. The project involves data preprocessing, feature extraction, model training, and evaluation to achieve the best possible performance.

## Features
- Data Preprocessing: Cleaning and preparing the SMS dataset, handling missing values, and converting text data into numerical format.
- Feature Extraction: Using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) to transform text data into feature vectors.
- Model Training: Implementing and training various machine learning models such as Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) to classify messages.
- Model Evaluation: Assessing the performance of the models using metrics like accuracy, precision, recall, and F1-score to ensure the classifier's effectiveness.
- Spam Detection: The final model predicts whether an incoming SMS is spam or ham.

## Tools and Technologies
- Programming Language: Python
- Libraries: pandas, numpy, scikit-learn, NLTK, and others
- Models Used: Naive Bayes, Logistic Regression, SVM
- Data Source: spam.csv

## Installation
 1.Clone the repository:
```git clone https://github.com/Kritika2121/sms-spams-classifications.git```


 2.Navigate to the project directory:
 ```cd sms-spam-classification```
 
 3.Install the required dependencies:
 ```pip install -r requirements.txt```

 ## Usage
 1.Run the main script to train the model and classify SMS messages:
 ```python sms_spam_classifier.py```
 
2.You can test the model with custom SMS messages by modifying the test script:
```python test_sms.py```


# Results
The model achieves a high accuracy and effectively filters spam messages. Detailed performance metrics and visualizations are provided in the results folder.

# Future Enhancements
- Integrating the model into a real-time SMS filtering system.
- Exploring deep learning techniques like LSTM for better accuracy.
- Expanding the dataset to include more diverse messages for improved generalization.

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for suggestions or bug reports.



 
