#imports
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#######
print "Starting...."
newsgroups_train = fetch_20newsgroups(subset='train')
print len(newsgroups_train.data)

vectorizer = CountVectorizer(stop_words = 'english')
transformer = TfidfTransformer()
##fit
X_train_data = newsgroups_train.data
y_train_data = newsgroups_train.target

X_train_counts = vectorizer.fit_transform(X_train_data)
X_train = transformer.fit_transform(X_train_counts)

time_starting = time.time()
nb_clf = MultinomialNB(alpha = 0.01).fit(X_train, y_train_data)
time_ending_nb = time.time() - time_starting

time_starting = time.time()
svm_clf = svm.SVC(kernel=cosine_similarity)
svm_clf.fit(X_train, newsgroups_train.target)
time_ending_svm = time.time() - time_starting
##predict
newsgroups_test = fetch_20newsgroups(subset = 'test')

X_test_data = newsgroups_test.data
y_test_data = newsgroups_test.target

X_test_count = vectorizer.transform(X_test_data)
X_test = transformer.transform(X_test_count)
y_predict_nb_train = nb_clf.predict(X_train)
y_predict_svm_train = svm_clf.predict(X_train)
y_predict_nb = nb_clf.predict(X_test)
y_predict_svm = svm_clf.predict(X_test)

accuracy_nb_train = accuracy_score(y_train_data, y_predict_nb_train)
accuracy_svm_train = accuracy_score(y_train_data, y_predict_svm_train)
accuracy1_nb = accuracy_score(y_test_data, y_predict_nb)
accuracy1_svm = accuracy_score(y_test_data, y_predict_svm)
print "============= Accuracy: ============= "
print "Naive Bayes: "
print "Accuracy on training data: " , accuracy_nb_train , " Accuracy on testing data: " ,accuracy1_nb
print "SVM with cosine similarity kernel: "
print "Accuracy on training data: " ,accuracy_svm_train, " Accuracy on testing data: ", accuracy1_svm

precision_nb_train = precision_score(y_train_data, y_predict_nb_train, average = "macro")
precision_svm_train = precision_score(y_train_data, y_predict_svm_train, average = "macro")
precision_nb = precision_score(y_test_data, y_predict_nb, average = "macro")
precision_svm = precision_score(y_test_data, y_predict_svm, average = "macro")
print "============= Precision: ============= "
print "Naive Bayes: "
print "Precision on training data: " ,precision_nb_train ," Precision on testing data: " ,precision_nb
print "SVM with cosine similarity kernel: "
print "Precision on training data: " ,precision_svm_train ," Precision on testing data: " ,precision_svm

recall_nb_train = recall_score(y_train_data, y_predict_nb_train, average = "macro")
recall_svm_train = recall_score(y_train_data, y_predict_svm_train, average = "macro")
recall_nb = recall_score(y_test_data, y_predict_nb, average = "macro")
recall_svm = recall_score(y_test_data, y_predict_svm, average = "macro")
print "============= Recall: ============="
print "Naive Bayes: "
print "Recall on training data: " ,recall_nb_train ," Recall on testing data: " ,recall_nb
print "SVM with cosine similarity kernel: "
print "Recall on training data: " ,recall_svm_train ," Recall on testing data: " ,recall_svm

print "============= Training Time: ============= "
print "Naive Bayes: "
print  time_ending_nb 
print "SVM with cosine similarity kernel: "
print  time_ending_svm