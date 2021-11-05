
1. JSON data source : news_data.json

2. CSV file : news_st.csv ( Exported news articles from MYSQL DB : news.articles (Table ))

    News articles are streamed via KAFKA producer ( Publisher ) consumed via KAFKA ( Consumer) 

    1. RAPIDAPI       .......................  : api_producer.py , api_consumer.py 

    2. STATIC News Data Source ...  : static_producer.py , static_consumer.py



