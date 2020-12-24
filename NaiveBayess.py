import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn import metrics
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize




ps = PorterStemmer()

jokesBody = []
jokesCategory = []
freq = {}

with open("stupidstuff.json") as f:
  data1 = json.load(f)


with open("wocka.json") as s:
  data2 = json.load(s)



#Preprossesong

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
        if joke.get("category") == "Crazy Jokes" or joke.get("category") == "Deep Thoughts" or joke.get("category") == "Ethnic Jokes" or joke.get("category") == "Aviation" or joke.get("category") == "State Jokes" or joke.get("category") == "Old Age": 
            joke["category"] = "Other / Misc"
        """
        b = "" 
        for token in word_tokenize(joke.get("body")):
            b += " " + ps.stem(token)
        """     
        jokesBody.append(joke.get("body"))
        #jokesBody.append(b)
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
        if joke.get("category") == "Crazy Jokes" or joke.get("category") == "Deep Thoughts" or joke.get("category") == "Ethnic Jokes" or joke.get("category") == "Aviation" or joke.get("category") == "State Jokes" or joke.get("category") == "Old Age":
            joke["category"] = "Other / Misc"
        
        jokesBody.append(joke.get("body"))
        #jokesBody.append(b)
        jokesCategory.append(joke.get("category"))

        if joke["category"] in freq.keys():
            freq[joke["category"]] += 1
        else:
            freq[joke["category"]] = 0


"""
for key, value in freq.items():
    print("Key :", key, "\n frequency : ", value)
"""



X_train, X_test, y_train, y_test = train_test_split(jokesBody, jokesCategory,test_size=0.2, random_state=0)


print("X_train dimension = ", len(X_train))
print("X_test dimension = ", len(X_test))
print("y_train dimension = ", len(y_train))
print("y_test dimension = ", len(y_test))



# FTiaxnw ta tf-idf
vectorizer = CountVectorizer(max_df=0.85, stop_words="english", ngram_range=(2,2))
X = vectorizer.fit_transform(X_train)

tfidf_Transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
X_train_tfidf = tfidf_Transformer.fit_transform(X)


#Ekpaideush tou modelou
clf = MultinomialNB().fit(X_train_tfidf, y_train)


#print(metrics.accuracy_score(y_test, clf.predict( vectorizer.transform(X_test))))

# Metrics
print(metrics.classification_report(y_test, clf.predict( vectorizer.transform(X_test))))

#print(vectorizer.vocabulary_) # to vocabulary
