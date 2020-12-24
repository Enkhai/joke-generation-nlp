import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer


"""
digits_df = datasets.load_digits()
print('Digits dataset structure= ', dir(digits_df))
print('Data shape= ', digits_df.data.shape)
print('Data conatins pixel representation of each image, \n', digits_df.data)


X = digits_df.data
y = digits_df.target
"""



jokesBody = []
jokesCategory = []
freq = {}

with open("stupidstuff.json") as f:
  data1 = json.load(f)


with open("wocka.json") as s:
  data2 = json.load(s)


#Preprocessing
for joke in data1:
    if joke.get("body") != "" and joke.get("category") != "Redneck" and joke.get("category") != "English" and joke.get("category") != "Music":  
        if joke.get("category") == "Men" or joke.get("category") == "Women":
            joke["category"] = "Men / Women"
        if joke.get("category") == "Political":
            joke["category"] = "News / Politics"
        if joke.get("category") == "Computers":
            joke["category"] = "Science"
        if joke.get("category") == "Children":
            joke["category"] = "Family, Parents"
        if joke.get("category") == "Idiots":
            joke["category"] = "Insults"
        if joke.get("category") == "Crazy Jokes" or joke.get("category") == "Deep Thoughts" or joke.get("category") == "Ethnic Jokes" or joke.get("category") == "Aviation" or joke.get("category") == "State Jokes":
            joke["category"] = "Other / Misc"

        jokesBody.append(joke.get("body"))
        jokesCategory.append(joke.get("category"))
        
        if joke["category"]  in freq.keys():
            freq[joke["category"]] += 1
        else:
            freq[joke["category"]] = 0


for joke in data2:
    if joke.get("body") != "" and joke.get("category") != "Redneck" and joke.get("category") != "English" and joke.get("category") != "Music":
        if joke.get("category") == "Animal":
            joke["category"] = "Animals"
        if joke.get("category") == "Bar":
            joke["category"] = "Bar Jokes"
        if joke.get("category") == "Blond" or joke.get("category") == "Blonde":
            joke["category"] = "Blonde Jokes"
        if joke.get("category") == "Lawyer":
            joke["category"] = "Lawyers"
        if joke.get("category") == "Yo Momma":
            joke["category"] = "Yo Mama"
        if joke.get("category") == "Children":
            joke["category"] = "Family, Parents"
        if joke.get("category") == "Lightbulb":
            joke["category"] = "Light Bulbs"
        if joke.get("category") == "Crazy Jokes" or joke.get("category") == "Deep Thoughts" or joke.get("category") == "Ethnic Jokes" or joke.get("category") == "Aviation" or joke.get("category") == "State Jokes":
            joke["category"] = "Other / Misc"

        jokesBody.append(joke.get("body"))
        jokesCategory.append(joke.get("category"))

        if joke["category"] in freq.keys():
            freq[joke["category"]] += 1
        else:
            freq[joke["category"]] = 0








X_train, X_test, y_train, y_test = train_test_split(jokesBody, jokesCategory,test_size=0.2, random_state=0)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2, random_state = 1)
"""
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)
"""

#Dhmiourgw ta tf-idf
vectorizer = CountVectorizer(max_df=0.85, stop_words="english")
X = vectorizer.fit_transform(X_train)
#print(vectorizer.get_feature_names())

tfidf_Transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
X_train_tfidf = tfidf_Transformer.fit_transform(X)






#Ekpaideusei modelou
lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
lm.fit(X_train_tfidf, y_train)

#print('Predicted value is =', lm.predict([X_test[200]]))

#print('Actual value from test data is %s and corresponding image is as below' % (y_test[200]) )

#print("Model score : ", lm.score(X_test, y_test))

# Deixnw metrics
print(metrics.classification_report(y_test, lm.predict(vectorizer.transform(X_test))))
