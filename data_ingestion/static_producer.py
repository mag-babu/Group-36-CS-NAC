from static_reads import static_news_reads

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

while True:

    news_articles = static_news_reads()

    for news_article in news_articles:                  

        news_article = json.dumps(news_article).encode('utf-8')
        print(news_article)

        prod.send(TOPIC,value=news_article)

        print('Articles Published ..')
        sleep(random.randint(2,4))


