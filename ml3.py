# Modified from:
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

import csv, re, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_selection import SelectKBest, chi2
from optparse import OptionParser
from time import time
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import density
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import numpy as np
import pylab as pl
from sklearn import metrics


#Loading Data

print('Loading training data')
cityName = []
cityCode =[]
countryCode = []

with open('../handout/training.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    # this will iterate over each row and get the cell(s)
    for row in reader:
        cityName.append(row[0])
        cityCode.append(row[1])
        countryCode.append(row[2])

print('Loading testing data')
cityNameTest = []
cityCodeTest =[]
countryCodeTest = []

with open('../handout/testing.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    # this will iterate over each row and get the cell(s)
    for row in reader:
        cityNameTest.append(row[0])

"""

Training with libraries

"""

### Parser using external options

op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
#op.add_option("--chi2_select",
#              action="store", type="int", dest="select_chi2",
#              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
op.add_option("--cities",
              action="store_true",
              help="Classify cities")
op.add_option("--countries",
              action="store_true",
              help="Classify countries")

(opts, args) = op.parse_args()

categories = None 
remove = ()

## Getting data

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)
#data_train.items()[1][0]: data
#data_train.items()[2][0]: target 
#data_train.items()[3][0]: target names
#data_train.items()[4][0]: filenames

#data_test = fetch_20newsgroups(subset='test', categories=categories,
#                               shuffle=True, random_state=42,
#                               remove=remove)

#categories = data_train.target_names    # for case categories == None

#len(data_train.target) : 11314
#y_train, y_test = data_train.target, data_test.target
trainSize = 8*(len(countryCode)-1)/10
if opts.countries:
        y_train = countryCode[:trainSize]
        y_valid = countryCode[trainSize:]

if opts.cities:
        y_train = cityCode[:trainSize]
        y_valid = cityCode[trainSize:]


print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
        #vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
        #                           n_features=opts.n_features)
        #X_train = vectorizer.transform(data_train.data)
        vectorizer = HashingVectorizer(non_negative=True)
        X_train = vectorizer.transform(cityName[:trainSize])
else:
        #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
        #X_train = vectorizer.fit_transform(data_train.data)
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
        X_train = vectorizer.fit_transform(cityName[:trainSize])
duration = time() - t0
print("done in %fs" % duration)
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the validation dataset using the same vectorizer")
t0 = time()
X_valid = vectorizer.transform(cityName[trainSize:])
#X_test = vectorizer.transform(cityNameTest)
duration = time() - t0
print("done in %fs" % duration)
print("n_samples: %d, n_features: %d" % X_valid.shape)
print()


"""
Selecting best k features
k: integer 

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_valid, y_train)
    X_valid = ch2.transform(X_valid)
    print("done in %fs" % (time() - t0))
    print()
"""

###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_valid)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_valid, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_valid, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_valid, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12,8))
pl.title("Score")
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)

pl.show()


"""
This is the an awesome Spelling Corrector
Now it's working using only the training set
The idea is to use only the words that are more frequent and assume that outliers 
are missspellings
we need to get rid of outliers(missspelled words) and then correct the data set
For this we can sort the words by frenquence, remove the less common for the 
dictionary and correct all the training set
"""

alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-'

def edits1(word):
   #all possible splits of a word
   #word[i] ith character in the word
   #word[:i] first ith characters
   #word[i:] deletes first ith characters
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   #deletes one missing character
   deletes    = [a + b[1:] for a, b in splits if b]
   #mixing characters
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   #replacing each character by one in the alphabeth
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   #adding one character
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   #returning a set of ... 
   # these are all possible misspellings
   return set(deletes + transposes + replaces + inserts)

#possible modified words
def known_edits2(word):
#Q) Nested loop???
#explain syntax a for b in B for a in A if a in NWORDS
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

#possible existing words?
def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

"""
Leo: This is a manual training
"""
# making everything lower case
# [a-z] matching characters between a and z 
# finding groups of words
# this splits the words taking 
# numbers from 0 to 9
# characters from a-z 
# '-' as a character to
# def words(text): return re.findall('[0-9a-z\-]+', text.lower()) 
def words(namesList): return [re.findall('[0-9a-z\-]+', name.lower()) for name in namesList]

def trainName(Names):
    #model is a dictionary 
    #lambda function returns 1
    model = collections.defaultdict(lambda: 1)
    for name in Names:
        for word in name:
                model[word] += 1
                
    return model

def trainCode(Codes):
    #model is a dictionary 
    #lambda function returns 1
    model = collections.defaultdict(lambda: 1)
    for code in Codes:
            model[code] += 1
                
    return model


#this creates the dictionary 
#this is a file with lots of words inside
NWORDS = trainName(words(cityName))
NCITIES = trainCode(cityCode)
NCOUNTRIES = trainCode(countryCode)

#nWords = count all words
#nCities = count all cities
totalWords = len(NWORDS)
totalCities = len(NCITIES)
#totalCountries = len(NCOUNTRIES)

#Training

#for country in NCOUNTRIES
#count cities inside a country (nCitiesInCountry)
"""
Leo: We could also use a sparse matrix
"""

d=collections.defaultdict(collections.defaultdict)
for country in NCOUNTRIES:
	for city in NCITIES:
		d[country][city] = 0

it=0
for country in countryCode:
        city=cityCode[it]
        d[country][city] = 1
        it+=1

nCitiesInCountry = collections.defaultdict(lambda:0)
prior = collections.defaultdict(lambda:0)
for country in NCOUNTRIES:
        for city in NCITIES:
                if d[country][city]==1:
                        nCitiesInCountry[country] += 1

#prior[country] = nCitiesInCountry/totalCities
for country in NCOUNTRIES:
        prior[country] = float(nCitiesInCountry[country])/float(totalCities);

#print('#Cities per Country',d)

#print('prior',prior)

#creaty a dictionary for all cities in a country: dictionaryCountry 
condProb = collections.defaultdict(collections.defaultdict)
for country in NCOUNTRIES:
        for word in NWORDS:
                condProb[word][country]=0

#       for each word in NWORDS
        #for word in NWORDS
#               count words for each country : wordsDictionaryCountry
                #wordsDictionaryCountry = collections.defaultdict()
#               for each word in NWORDS
#                       condProb[word][country] = (wordsDictionaryCountry + 1)/sum(wordsDictionaryCountry'+1)

#return prior, condProb


#APPLY
#for each country 
#       score[country] = log prior[country]

#       for each word in name
#               score[c] += log conProb[word][country]

#return argmax score[c]

"""
In case we need to sort our dictionary
"""
#model = {'a':2, 'b':5, 'c':3}
#model {'a': 2, 'c': 3, 'b': 5}
#sorted(model.values()) [2, 3, 5] 
#sorted(model, key=model.get) ['a', 'c', 'b']
#sorted(model.items(), key=lambda x:x[1]) [('a', 2), ('c', 3), ('b', 5)]
#this is a list of sorted words
#sortedWords = sorted(NWORDS,key=NWORDS.get)
#this is a sorted dictionary
sortedWords = sorted(NWORDS.items(), key=lambda x:x[1])
