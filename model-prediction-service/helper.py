# database helper util
import pandas as pd 
import re
from connecttomysqldb import *

#pyspark modules
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import isnan, when, count, col, split, udf, sum, max, concat

from lime import lime_text
from lime.lime_text import LimeTextExplainer

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')

wnl = WordNetLemmatizer()

# get all records
def get_all(conn):
    # Get the park connection.
    spark = SparkSession.builder.config("spark.jars","jars/mysql-connector-java-8.0.25.jar").master("local").appName("PySpark_MySQL_test").getOrCreate()
    df = spark.read.format("jdbc").option("url","jdbc:mysql://localhost:3306/news").option("driver", "com.mysql.cj.jdbc.Driver").option("dbtable", "articles").option("user", "mag").option("password", "mag123").load() 
    print("Mysql Connected")
    print(df.show(3))
    return df


# driver code
if __name__ == '__main__':
    # connect to database and get all data
    get_all(connect())



# get total number of records
def get_records_count(cursor):
    # execute the command
    cursor.execute('''SELECT * FROM articles;''')
    return len(cursor.fetchall())


# get by unq_serno
def get_by_unq_serno(cursor, eid):
    # sql query
    query = '''SELECT * FROM articles WHERE id = %s;'''
    # execute the command
    cursor.execute(query, [eid])
    return cursor.fetchone()

def text_preproc(pars):

    pars = re.sub('[a-zA-Z]', ' ', pars)
    pars = pars.lower()
    pars = pars.split()
    pars = [wnl.lemmatize(word) for word in pars if not word in stopwords]
    pars = ' '.join(pars)
    return pars
