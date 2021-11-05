import requests 
from bs4 import BeautifulSoup
from datetime import datetime

def hackernews_rss():
       article_list = []

       r = requests.get('https://news.ycombinator.com/rss')
       soup = BeautifulSoup(r.content)

       articles = soup.findAll('item')

       for a in articles:
           title = a.find('title').text
           link = a.find('link').text
           published_date = datetime.strptime(a.find('pubdate').text,'%a, %d %b %Y %H:%M:%S +%f')
           summary = a.find('comments').text
           topic = ''
           source=''
           news_id = 0
           article = {'title':title,'published_date':published_date.strftime('%d/%m/%Y'), 'summary':summary, 'topic':topic,'source':source,'source_link':link,'news_src':'rss','news_id':news_id} 
           article_list.append(article)
       return article_list

if __name__ == '__main__':
    hackernews_rss()
