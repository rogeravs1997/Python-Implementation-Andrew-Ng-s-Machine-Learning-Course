# USEFUL LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize #fmin_cg to train the linear regression
from sklearn.svm import SVC #SVM software
import re
import nltk
from nltk import word_tokenize
from stemming.porter2 import stem


# Loading the vocab with most common words used in spam emails, and building PythonDictionary with index:word and word:index 
vocab_list = np.loadtxt(r'D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex6\ex6\vocab.txt', dtype='str')
vocab_index_dict = {row[1]: int(row[0]) for row in vocab_list}
index_vocab_dict = {int(row[0]): row[1] for row in vocab_list}

# Loading data
spam_train = scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex6\ex6\spamTrain.mat')
spam_test = scipy.io.loadmat('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex6\ex6\spamTest.mat')

X_train = spam_train['X']
y_train = spam_train['y'].ravel()

X_test = spam_test['Xtest']
y_test = spam_test['ytest'].ravel()


# PREPROCESSING EMAILS
stemmer = nltk.PorterStemmer()

def process_email(email_contents, verbose=True):
    
    word_indices = []
    
    # Add code to strip headers here?
    email_contents = email_contents.lower()
    
    # Strip all HTML
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)
    
    # Handle Numbers
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)
    
    # Handle URLS
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)
    
    # Handle Email Addresses
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)
    
    # Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)
    
    # Handle punctuation and special ascii characters
    email_contents = re.sub(r'[@$/\\#,-:&*+=\[\]?!(){}\'\">_<;%]+', '', 
                            email_contents)    
    
    # Tokenize
    word_list = word_tokenize(email_contents)
    
    for i, word in enumerate(word_list):
        # Remove punctuation and non-alphanumeric characters.
        word = re.sub(r'[^a-zA-Z0-9]', '', word)
        
        # If remaining word length is zero, continue.
        if len(word) < 1:
            continue
          
        # Stem 
        try:
            word = stemmer.stem(word)
        except:
            continue                
        try:
            word_indices.append(vocab_index_dict[word])
        except:
            continue

    return word_indices



# DEFINING FUNCTION TO GET FEATURES FROM ANY EMAIL, RETURN A 1899 DIMENSION VECTOR WITH 0's FOR NON COINCIDENCE WITH VOCAB_LIST AND 1's FOR COINCIDENCES
def email_features(word_indices):
    features = np.zeros(len(index_vocab_dict.keys()))
    for index in word_indices:
        features[index - 1] = 1
    return features

# One example of an email before an after the preprocessing, UNCOMMENT and UNTAB TO SEE
    # email_contents_1 = open('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex6\ex6\emailSample1.txt', 'r').read()
    # word_indices_1 = process_email(email_contents_1)
    # print(email_contents_1)
    # print(word_indices_1)
    # features_1 = email_features(word_indices_1)


# TRAINING THE SVM WITH LINEAR KERNEL 
spam_classificator = SVC(C=0.1, kernel='linear')
spam_classificator.fit(X_train, y_train)

# To show acurracy of the model uncomment the next line.
# print ('training accuracy: {}%'.format( spam_classificator.score(X_train, y_train)*100))
# print ('test accuracy: {}%'.format( spam_classificator.score(X_test, y_test)*100))

# Function to show the features (in this case Words) with larger weight to classidfy some sample as Positive
def topWeights(): 
    weights = spam_classificator.coef_.reshape(-1)
    sorted_indices = np.argsort(weights)[::-1]
    print ('top predictors of spam:')
    for index in sorted_indices[:20]:
        print ('%10s  %.2f' % (vocab_list[index][1], weights[index]))

# topWeights()

# FUINCTION TO CLASSIFY ANY EMAIL
def email_classifier(file_name):
    with open(file_name, 'r') as f:
        email_contents = f.read()

    word_indices = process_email(email_contents, verbose=False)
    features = email_features(word_indices)

    print ("CLASSIFICATION: ", "SPAM" if spam_classificator.predict(features.reshape(1, -1))[0] else "NOT SPAM")
    
    
# email_classifier('D:\Desktop\MACHINE LEARNING\Models\machine-learning-ex6\ex6\spamSample1.txt')
    
    
