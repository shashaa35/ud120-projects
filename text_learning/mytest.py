from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
s1 ="Hi hello hi hell oo ooo oooo oo"
s2 = "hello ji ki kr rahe ho oo"
str_list = [s1,s2]
bag = vectorizer.fit(str_list)
print(bag)
bag = vectorizer.transform(str_list)
print(bag)
print(vectorizer.vocabulary_.get("ooo"))