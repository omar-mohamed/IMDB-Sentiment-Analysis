# IMDB-Sentiment-Analysis
Sentiment analysis on IMDB movie reviews using Rnns and Lstms in Tensorflow.

# Problem:

Given a written review of a movie, the system should determine the sentiment behind the review and classify it into either good or bad.


# Dataset:

IMDB Movie Reviews Dataset: The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

### File descriptions:
- labeledTrainData - The labeled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review.  
- testData - The test set. The tab-delimited file has a header row followed by 25,000 rows containing an id and text for each review. Your task is to predict the sentiment for each one. 
- unlabeledTrainData - An extra training set with no labels. The tab-delimited file has a header row followed by 50,000 rows containing an id and text for each review. 
- sampleSubmission - A comma-delimited sample submission file in the correct format.

### Data fields:
- id - Unique ID of each review
- sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
- review - Text of the review

Note: Most of our work will be on the 25k reviews contained in 'labeledTrainData' which is already uploaded in this repo.

Link: [IMDB_Movie_Reviews_Dataset](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)

### Dataset Splitting:
- 20k reviews from 'labeledTrainData' were used for training.
- 1k reviews from 'labeledTrainData' were used for validation.
- 4k reviews from 'labeledTrainData' were used for testing.

# Preprocessing:

- **Clean text**: by removing extra spaces, numbers, making all letters lowercase, and removing special characters
- **Removing stop words**: removing some common stop words from the text that does not add much information regarding the sentiment.       (The stop words are dataset/stopwords.txt)
- **Tokenize the reviews**: by giving each word present in the dataset a unique number and replace the words with these numbers. 
  (When predicting a new review, if a word does not exist in the dataset and does not have a token it will be ignored)
- **Pad the sequences**: by unifying the length of every review (ex 200). Shorter reviews will be padded with 0, and longer ones will be    cut.


# Training Method:

- Used Recurrent neural nets with LSTM.

![cntk106a_model_s3](https://user-images.githubusercontent.com/6074821/46958649-9a799380-d09a-11e8-98c5-db285e002eed.png)

- Used Dropout regularization
- Used early stopping
- Used learning rate decay
- Used gradient clipping 
- Used Adam optimizer on training batches


# Results (ongoing):

# Usage:

Here we will discuss how to run the project, some implementational details, and variables to tweak to increase accuracy

### PreprocessData.py:
This is the first script you should run to make a .pickle file of the dataset. It cleans the data, removes stop words, tokinze, and     pad the sequences.

Variables that can be easily tweaked are:
- Stopwords (in dataset/stopwords.txt)
- Padding length 
- Size of training, validation, and test data

# Environment Used:
- Python 3.6.1
- Tensorflow 1.9
