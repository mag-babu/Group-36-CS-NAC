from api_reads import api_news_reads

from kafka import KafkaProducer

import pandas as pdf
import sys
import json
import random
from time import sleep

BROKER = 'localhost:9092'                                                                                               
TOPIC = 'articles' 

try:
    prod  = KafkaProducer(bootstrap_servers=BROKER)

except Exception as e:    
                                                                                              
    print(f"ERROR : {e}")    
                                                                                         
    sys.exit(1)           

last_news_info = {'title':''}

while True:

    news_articles = api_news_reads()

    if last_news_info['title'] == news_articles[0] ['title']:

            print('No (New) Articles Published ..')

            sleep(600)

    else:

        for news_article in news_articles:

                   if news_article != last_news_info['title']:

                         news_article = json.dumps(news_article).encode('utf-8')
                         print(news_article)

                         prod.send(TOPIC,value=news_article)

                         print('Articles Published ..')

                         sleep(random.randint(2,4))

                   else:

                       break

    last_news_info = news_articles[0]
            
