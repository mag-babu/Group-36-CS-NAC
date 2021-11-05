import mysql.connector

from json import loads
import json 

from time import sleep

from kafka import KafkaConsumer

consumer = KafkaConsumer('articles',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group-id',
    value_deserializer=lambda x: loads(x.decode('utf-8'))
)

mydb = mysql.connector.connect(
    host="localhost",
    user="mag",
    password="mag123",
    database="news"
)

mycursor = mydb.cursor()

for article in consumer:

    # mycursor.execute('select title from news.articles')
    # print(mycursor.fetchall())
   
try:

    article_info = article.value
    print(article_info)
    info=article_info
    sql_str = "INSERT INTO news.articles (title, date_time,summary,topic,data_source,news_source) VALUES (%s, %s,%s,%s,%s,%s)"
    values = (info['title'],info['published_date'],info['summary'],info['topic'], info['news_src'],info['source'])
    mycursor.execute(sql_str, values)
    mydb.commit();   
    print('Information Saved Sucessfully ...')
    

except Exception as e:    
                                                                                              
    print(f"ERROR : {e}")                                                                                        
    # sys.exit(1)           
