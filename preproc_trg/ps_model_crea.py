# sklearn Modules

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score

import pandas as pd
import pickle
import re

import string
import warnings
import numpy as np

from lime import lime_text
from lime.lime_text import LimeTextExplainer

# pyspark Modules

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import isnan, when, count, col, split, udf, sum, max,concat

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')

spark = SparkSession.builder.config("spark.jars", "jars/mysql-connector-java-8.0.25.jar") \
    .master("local").appName("PySpark_MySQL_test").getOrCreate()

df = spark.read.format("jdbc").option("url", "jdbc:mysql://localhost:3306/news") \
    .option("driver", "com.mysql.cj.jdbc.Driver").option("dbtable", "articles") \
    .option("user", "mag").option("password", "mag123").load()

print("Mysql Connected")

sdf1=df.select("summary","title","topic").filter("topic is not  null and title is not null").distinct();

print (df.show(3))

wnl = WordNetLemmatizer()

def text_preproc(pars):

    pars = re.sub('[^a-zA-Z]', ' ', pars)
    pars= pars.lower()
    pars = pars.split()
    pars = [wnl.lemmatize(word) for word in pars if not word in stopwords]
    pars = ' '.join(pars)
    return pars

text_preproc_udf = udf(lambda t: text_preproc(t),StringType())

sdf3=sdf1.select(concat("summary","title").alias("newstxt"),"topic")
sdf4 = sdf3.select("newstxt","topic").withColumn('news', text_preproc_udf(col("newstxt")))

rawdf = sdf4.toPandas()

print("** Reached Pandas **")

X = rawdf[['news']]
y = rawdf['topic']

encode = LabelEncoder()

y = encode.fit_transform(y)

print("** Encode Fit reached ** ")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

x_train=x_train.reset_index().drop(['index'],axis=1)
x_test=x_test.reset_index().drop(['index'],axis=1)

vectorize = TfidfVectorizer()

x_trn_tokens = vectorize.fit_transform(x_train['news'].values).toarray()

print("** Vectorize Fit reached ** ")

x_tst_tokens =vectorize.transform(x_test['news'].values).toarray()

clf_mnb=MultinomialNB(alpha=.01)

clf_mnb.fit(x_trn_tokens, y_train )

print("** MNB Fit reached ** ")

pred = clf_mnb.predict(x_tst_tokens)

# Print Scores

print("** Score Card ( Start ) ** ")

print(cross_val_score(clf_mnb,x_trn_tokens, y_train, cv=2, scoring="accuracy" ))
print(classification_report(y_test, pred))

print(f1_score(y_test, pred, average='weighted'))

print("** Score Card ( End ) ** ")

# Saving Py Spark Trained Models

print("MNB/Vectorize/Encode Models ( Start ) ")

pickle.dump(clf_mnb,open('models/psp_mnb.pkl','wb'))

pickle.dump(vectorize,open('models/psp_vector.pkl','wb'))
pickle.dump(encode,open('models/psp_encode.pkl','wb'))

print("MNB/Vectorize/Encode Models ( end ) ")

pipe_line = make_pipeline(vectorize, clf_mnb)

print(" Explainer ( Start ) ")

explainer = LimeTextExplainer(class_names=encode.classes_)
idx=8
class_names=encode.classes_
exp = explainer.explain_instance(x_test['news'][idx], pipe_line.predict_proba, num_features=3, top_labels=2)
print(exp.available_labels())

print(" Explainer ( End ) ")

