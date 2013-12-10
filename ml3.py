import csv, re, collections
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading Data
print('Loading csv data')

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

print('#Cities per Country',d)

print('prior',prior)

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

Training with libraries

# split a training set and a test set

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

categories = data_train.target_names    # for case categories == None

y_train, y_test = data_train.target, data_test.target

if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
"""
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)

ch2 = SelectKBest(chi2, k=opts.select_chi2)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)



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
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
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
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


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
This is how we sort our dictionary
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
