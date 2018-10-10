import keras
import nltk
import pandas as pd
import re
import codecs
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np

# taken from https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e
# code: http://nbviewer.jupyter.org/github/hundredblocks/concrete_NLP_tutorial/blob/583d1a0f399ddff78cd928e07e35f4cc4b12f9bc/NLP_notebook.ipynb

#input_file = codecs.open("socialmedia_relevant_cols.csv", "r",encoding='utf-8', errors='replace')
#output_file = open("socialmedia_relevant_cols_clean.csv", "w")

def sanitize_characters(raw, clean):    
    for line in raw:
        clean.write(line)
#sanitize_characters(input_file, output_file)
questions = pd.read_csv("socialmedia_relevant_cols_clean.csv", usecols=['text', 'choose_one', 'class_label'])
questions.columns=['text', 'choose_one', 'class_label']
#print(questions.head())
#print(questions.tail())
#print(questions.describe())

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

#questions = standardize_text(questions, "text")

#questions.to_csv("clean_data.csv")
#print(questions.head())
clean_questions = pd.read_csv('clean_data.csv')
#print(clean_questions.groupby("class_label").count())

tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
#print(clean_questions.head())

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
VOCAB = sorted(list(set(all_words)))
#print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
#print("Max sentence length is %s" % max(sentence_lengths))

fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Sentence length')
plt.ylabel('Number of sentences')
plt.hist(sentence_lengths)
#plt.show()

def cv(train, test):
    count_vectorizer = CountVectorizer()
    '''
    X_train_counts = a 2d array where each element is how many times each word in the set of all words appears
    like so:
    [[0 1 1 1 0 0 1 0 1]
    [0 2 0 1 0 1 1 0 1]
    [1 0 0 1 1 0 1 1 1]
    [0 1 1 1 0 0 1 0 1]]

    bag of words model- ignore order of words in sentences
    '''
    # The fit part fits the vocabulary to the test data
    X_train_counts = count_vectorizer.fit_transform(train)
    # don't have to call fit twice because it's already fitted
    X_test_counts = count_vectorizer.transform(test)

    return X_train_counts, X_test_counts

list_corpus = clean_questions["text"].tolist()
list_labels = clean_questions["class_label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, 
                                                                                random_state=40)

X_train_counts, X_test_counts = cv(X_train, X_test)

def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    '''
    dimenstionality reduction: https://www.geeksforgeeks.org/dimensionality-reduction/
    = reducing number of overlapping variables
    truncated SVD = truncated single value decomposition or lsa (latent semantic analysis)

    PCA (principal component analysis) centers data before computing SVD, 
    therefore truncated svd better for sparse matrices

    http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/
    first step in lsa is tf-idf (term frequency- inverse document frequency)
    count number of times word appears in document, normalize word counts by frequency of word in overall collection
    next, use SVD to perform dimensionality reduction

    gensim- topic modeling: https://radimrehurek.com/gensim/ 
    '''
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange','blue','blue']
    if plot:
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Irrelevant')
        green_patch = mpatches.Patch(color='blue', label='Disaster')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


fig = plt.figure(figsize=(16, 16))          
plot_LSA(X_train_counts, y_train)
#plt.show()

clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)

y_predicted_counts = clf.predict(X_test_counts)

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt

cm = confusion_matrix(y_test, y_predicted_counts)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Irrelevant','Disaster','Unsure'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)