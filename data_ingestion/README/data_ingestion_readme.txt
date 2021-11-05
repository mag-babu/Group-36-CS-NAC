DB Installation Steps : -
----------------------------
sudo mysql

CREATE USER 'user'@'localhost' IDENTIFIED BY 'user123';

create database news;
use news;
create table articles(title varchar(200),date_time varchar(20),summary varchar(1000),topic varchar(50),data_source varchar(30),news_source varchar(200),PRIMARY KEY (title));

describe articles; 

+-------------+---------------+------+-----+---------+-------+
| Field       | Type          | Null | Key | Default | Extra |
+-------------+---------------+------+-----+---------+-------+
| title       | varchar(200)  | NO   | PRI | NULL    |       |
| date_time   | varchar(20)   | YES  |     | NULL    |       |
| summary     | varchar(1000) | YES  |     | NULL    |       |
| topic       | varchar(50)   | YES  |     | NULL    |       |
| data_source | varchar(30)   | YES  |     | NULL    |       |
| news_source | varchar(200)  | YES  |     | NULL    |       |
+-------------+---------------+------+-----+---------+-------+

select topic,count(*) from articles group by topic;

1. For Rapid API 
---------------------

	1.1 api_producer.py   ( Publisher )
	1.2 api_consumer.py ( Consumer) 

		news articles gathered stored in back end mysql 

		db DB : news Table : articles

2. For Static News Data Source
---------------------------------------

	1.1 static_producer.py   ( Publisher )
	1.2 static_consumer.py ( Consumer) 

		news articles gathered stored in back end mysq

	                DB : news Table : articles