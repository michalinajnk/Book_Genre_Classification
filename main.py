import os
import string
import random
from collections import defaultdict
import itertools as it
import nltk
import numpy as np
import pandas as pd

from nltk import word_tokenize, FreqDist, metrics
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import pickle
from nltk.stem import WordNetLemmatizer

# import nltk
# nltk.download('stopwords')
from sklearn.svm import SVC

stop_words = set(stopwords.words('english'))
stop_words.add('said')
stop_words.add('mr')

FILE = f'C:/Users/Michalina/OneDrive/Pulpit/booksummaries.txt'
PROCESSED_FILE = f'C:/Users/Michalina/OneDrive/Pulpit/preprocessed_books.txt'
PROCESSED_FILE_TEST = f'C:/Users/Michalina/OneDrive/Pulpit/preprocessed_books_test.txt'
LABELS = ["Mystery", "Fiction", "Fantasy", "Novel", "History"]



def create_data_set():
	with open(PROCESSED_FILE, 'w', encoding='utf8') as outfile:
		with open(FILE, 'r', encoding='utf 8') as file:
			books = get_books_from_splitted_text(file.read())
			for book in books:
				outfile.write('%s\t%s\n' % (str(book[0]), str(book[1])))


def create_data_set_test():
	with open(PROCESSED_FILE_TEST, 'w', encoding='utf8') as outfile:
		with open(FILE, 'r', encoding='utf 8') as file:
			books = get_books_from_splitted_text_test(file.read())
			for book in books:
				outfile.write('%s\t%s\n' % (str(book[0]), str(book[1])))


def read_books_from_file(filename):
	books = []
	with open(filename, 'r', encoding='utf 8') as datafile:
		for row in datafile:
			parts = row.split('\t')
			book = (parts[0], parts[1].strip())
			books.append(book)
	return books


def get_books_from_splitted_text(text):
	books = text.split('\n')
	all_books = []
	for book in books:
		bk = create_preprocessed_Book(book)
		if bk is not None:
			all_books.append(bk)
	return remove_books(all_books)


def get_books_from_splitted_text_test(text):
	books = text.split('\n')
	all_books = []
	for book in books:
		bk = create_Book(book)
		if bk is not None:
			all_books.append(bk)
	return remove_books(all_books)


def get_genre(genres):
	gen = _get_all_genres(genres)
	for genre in gen:
		if genre in LABELS:
			return genre


def _get_all_genres(genres):
	gen = []
	almost_book_genres = genres.split(": ")
	for genre in almost_book_genres:
		g = genre.split(", ")
		for word in g:
			word = word.replace('"', "")
			word = word.replace('}', "")
			word = word.replace('{', "")
			if word.isalpha():
				gen.append(word.strip())
	return gen


def create_preprocessed_Book(book):
	book_tokens = book.split("\t")
	book_tokens = book.split("\t")
	if len(book_tokens) >= 7 and get_genre(book_tokens[5]) is not None and book_tokens[6].strip() is not None:
		book = (get_genre(book_tokens[5]), prepocessor(book_tokens[6].strip()))
		return book
	else:
		return None


def create_Book(book):
	book_tokens = book.split("\t")
	book_tokens = book.split("\t")
	if len(book_tokens) >= 7 and get_genre(book_tokens[5]) is not None and book_tokens[6].strip() is not None:
		book = (get_genre(book_tokens[5]), book_tokens[6].strip())
		return book
	else:
		return None


def clean_text(text):  # remove puntcuation , fixed to lower case
	text = text.translate(str.maketrans('', '', string.punctuation))
	text = text.lower()
	return text


def remove_books(books):
	for book in books:
		if book[0] == "-" or book[1] == "-":
			books.remove(book)
	return books


def get_tokens(text):  # tokenize nd remove stopwords
	clean_text(text)
	lemmatizer = WordNetLemmatizer()
	tokens = word_tokenize(text)
	tokens = [lemmatizer.lemmatize(word) for word in tokens if not word in stopwords.words('english')]
	return tokens


def prepocessor(text):
	tokens = get_tokens(text)
	preprocessed = ' '.join(tokens)
	return preprocessed


def get_tokes_lemm(text):
	tokens = get_tokens(text)
	lemmatizer = WordNetLemmatizer()
	return [lemmatizer.lemmatize(word) for word in tokens]


def get_frequency_dist(docs):
	tokens = defaultdict(list)
	for doc in docs:
		doc_label = doc[0]
		doc_text = clean_text(doc[1])
		doc_tokens = get_tokens(doc_text)
		tokens[doc_label].extend(doc_tokens)

	for category_label, category_tokens in tokens.items():
		print(category_label)
		fd = FreqDist(category_tokens)
		print(fd.most_common(20))


def get_splits(docs, pivot_point):
	random.shuffle(docs)

	X_train = []  # training documents
	y_train = []  # corresponding training labels

	X_test = []  # test documents
	y_test = []  # corresponding test labels

	pivot = int(pivot_point * len(docs))

	for i in range(0, pivot):
		X_train.append(docs[i][1])
		y_train.append(docs[i][0])

	for i in range(pivot, len(docs)):
		X_test.append(docs[i][1])
		y_test.append(docs[i][0])

	return X_train, X_test, y_train, y_test


def evaluate_classifier_train(classifier, X_train, Y_train, cross_validation):
	assessment = cross_val_score(classifier, X_train, Y_train, cv=cross_validation)
	print(assessment)
	print(np.mean(assessment))


"""CONFUSION MATRIX
Accuracy = (TP + TN) /(TP + TN + FP + FN)
Precision = (TP) / (TP + FP) --> positive predicted value
Recall = (TP) / (TP + FN)  --> sensitivity of my model
F1 Score = (2 x Precision x Recall) / (Precision + Recall)
—  where TP is True Positive, FN is False Negative and likewise for the rest."""


def evaluate_classifier_test(title, classifier, vectorizer, X_test, y_test):
	X_test_tfidf = vectorizer.transform(X_test)
	y_pred = classifier.predict(X_test_tfidf)

	precision = metrics.precision_score(y_test, y_pred, average=None)
	recall = metrics.recall_score(y_test, y_pred, average=None)
	f1 = metrics.f1_score(y_test, y_pred, average=None)
	print("%s\t%f\t%f\t%f\n" % (title, precision[0], recall[0], f1[0]))


# CROSS VALIDATION
def train_classifier(classifier, vectorizer, X_train, X_test, y_train, y_test, title_test_classifier):
	dtm = vectorizer.fit_transform(X_train)
	fitted_classifier = classifier.fit(dtm, y_train)
	evaluate_classifier_train(fitted_classifier, X_train, y_train, cross_validation=5)
	evaluate_classifier_test(title_test_classifier, classifier, vectorizer, X_test, y_test)
	return fitted_classifier, vectorizer, X_test, y_test


def _get_params_combi(param_grid):
	all_combs = sorted(param_grid)
	params_combi = it.product(*(param_grid[key] for key in all_combs))
	return list(params_combi)


def save_estimator_to_file(file_title, classifier):
	_save_to_file(file_title, classifier)


def save_vectorizer_to_file(file_title, vectorizer):
	_save_to_file(file_title, vectorizer)


def _save_to_file(file_title, obj):
	clf_filename = file_title
	pickle.dump(obj, open(clf_filename, 'wb'))


"""cross validation used for search best  hiper params
n_jobs number of jobs run in parallel
verbose controls the number of"""


def get_MNB_with_theBest_hiperParams(cross_validation_k, X_train, y_train):
	vectorizer = CountVectorizer(stop_words='english',
	                             ngram_range=(1, 2),
	                             min_df=3, analyzer='word')

	dtm = vectorizer.fit_transform(X_train)

	mnb_model = MultinomialNB()
	param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}

	rt_Grid = RandomizedSearchCV(estimator=mnb_model, param_distributions=param_grid, cv=cross_validation_k, verbose=3,
	                             n_jobs=4)
	rt_Grid.fit(dtm, y_train)
	print("BEST PARAMS FOR MultinomialNB ARE: ")
	print(rt_Grid.best_params_)

	save_estimator_to_file('naive_bayes_classifier.pkl', rt_Grid.best_estimator_)
	save_vectorizer_to_file('count_vectorizer.pkl', vectorizer)

	return rt_Grid.best_estimator_, vectorizer


def get_SVM_with_theBest_hiperParams(cross_validation_k, X_train, y_train):
	vectorizer = CountVectorizer(stop_words='english',
	                             ngram_range=(1, 2),
	                             min_df=3, analyzer='word')

	dtm = vectorizer.fit_transform(X_train)

	svm_model = SVC(gamma='auto')
	param_grid = {'C': [0.1, 1, 10, 100],
	              'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

	rt_Grid = RandomizedSearchCV(estimator=svm_model, param_distributions=param_grid, cv=cross_validation_k, verbose=3,
	                             n_jobs=4)
	rt_Grid.fit(dtm, y_train)
	print("BEST PARAMS ARE: ")
	print(rt_Grid.best_params_)

	save_estimator_to_file('support_vector_classifier.pkl', rt_Grid.best_estimator_)
	save_vectorizer_to_file('count_vectorizer.pkl', vectorizer)

	return rt_Grid.best_estimator_, vectorizer


def classify(file_of_classifier, text):
	# load the classifier
	nb_clf = pickle.load(open(file_of_classifier, 'rb'))

	# vectorize the new text
	vec_filename = 'count_vectorizer.pkl'
	vectorizer = pickle.load(open(vec_filename, 'rb'))

	y_pred = nb_clf.predict(vectorizer.transform([text]))
	print('This text is %s.' % (y_pred[0]))


def convert_to_data_frame(books):
	df = pd.DataFrame(books, columns=['genre', 'text'])
	print(df)


def test_uncleaned_text(X_train, X_test, y_train, y_test):
	create_data_set_test()
	books_test = read_books_from_file(PROCESSED_FILE_TEST)
	get_frequency_dist(books_test)
	train_classifier(classifier=MultinomialNB(), vectorizer=CountVectorizer(stop_words='english',
	                                                                        ngram_range=(1, 2),
	                                                                        min_df=3, analyzer='word'), X_train=X_train,
	                 X_test=X_test, y_train=y_train, y_test=y_test, title_test_classifier="TESTING UNCLEANED TEXT")


def test_feature_selection_param_vect(X_train, X_test, y_train, y_test):
	param_grid = {'ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)], 'min_df': [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5],
	              'max_df': [0.01, 0.1, 0.3, 0.4, 0.5, 0.8, 1.0]}
	param_comb = _get_params_combi(param_grid)
	classifier = MultinomialNB()
	for (ngram_range, min_df, max_df) in param_comb:
		vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram_range, min_df=min_df, max_df=max_df,
		                             analyzer='word')
		title_test_classifier = 'Test Vectorizer params: n_gram_range: ' + ngram_range + ', min_df: ' + min_df \
		                        + ', max_df: ' + max_df

		train_classifier(classifier, vectorizer, X_train, X_test, y_train, y_test, title_test_classifier)


def test_sensitivity_to_hiperparams_SVM(X_train, X_test, y_train, y_test):
	classifier = SVC()
	vectorizer = CountVectorizer(stop_words='english',
	                             ngram_range=(1, 2),
	                             min_df=3, analyzer='word')

	dtm = vectorizer.fit_transform(X_train)


def test_sensitivity_to_hiperparams_MNB(X_train, X_test, y_train, y_test):
	classifier = MultinomialNB()
	vectorizer = CountVectorizer(stop_words='english',
	                             ngram_range=(1, 2),
	                             min_df=3, analyzer='word')

	dtm = vectorizer.fit_transform(X_train)


if __name__ == '__main__':
	# books = create_data_set()
	books = read_books_from_file(PROCESSED_FILE)
	convert_to_data_frame(books)
	# get_frequency_dist(books)

	X_train, X_test, y_train, y_test = get_splits(books, 0.80)
	best_estimatorSVM, vectorizer = get_SVM_with_theBest_hiperParams(cross_validation_k=2, X_train=X_train,
	                                                                 y_train=y_train)
	evaluate_classifier_test("Evaluation of SVM classifier", best_estimatorSVM, vectorizer, X_test, y_test)

	bestEstimatorMNB, vectorizer = get_MNB_with_theBest_hiperParams(cross_validation_k=2, X_train=X_train,
	                                                                y_train=y_train)
	evaluate_classifier_test("Evaluation of MNB classifier", bestEstimatorMNB, vectorizer, X_test, y_test)
