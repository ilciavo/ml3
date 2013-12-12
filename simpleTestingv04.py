import csv, re, collections
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_selection import SelectKBest, chi2
from optparse import OptionParser
from time import time
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
#from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import density
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import numpy as np
import pylab as pl
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer

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
        #cityCode.append(int(row[1]))
        #countryCode.append(int(row[2]))
        cityCode.append(row[1])
        countryCode.append(row[2])


print('Loading testing data')
cityNameTest = []

with open('../handout/validation.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    # this will iterate over each row and get the cell(s)
    for row in reader:
        cityNameTest.append(row[0])

###Training with libraries

categories = None 
remove = ()

X_train = cityName;

print('Creating the vectorizer and chosing a transform (from raw text to feature)')
vect= TfidfVectorizer(sublinear_tf=True, max_df=0.5)
#vect=CountVectorizer(min_n=1,max_n=2,max_features=1000);

X_train = vect.fit_transform(X_train)


cityClass = RidgeClassifier(tol=1e-7)
countryClass = RidgeClassifier(tol=1e-7)

print('Creating a classifier for cities')
cityClass.fit(X_train,cityCode)
print('Creating a classifier for countries')
countryClass.fit(X_train,countryCode)

print('testing the performance');

testCityNames = vect.transform(cityNameTest);

predictionsCity = countryClass.predict(testCityNames);
predictionsCountry = cityClass.predict(testCityNames);

with open('predictions.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        #for ind in range(0,len(predictionsCountry)):
        #        writer.writerow([str(predictionsCountry[ind]),str(predictionsCity[ind])])
        for predCountry,predCity in zip(predictionsCountry,predictionsCity):
                writer.writerow([predCountry,predCity])
