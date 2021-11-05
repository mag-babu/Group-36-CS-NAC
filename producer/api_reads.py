import requests
import json

def api_news_reads():

      url = "https://free-news.p.rapidapi.com/v1/search"

      qry = {"q":"news","lang":"en"}

      headers = {
        'x-rapidapi-key': "33735bb041mshcbff896cb9600d6p1771dajsn308a18f24fae",
        'x-rapidapi-host': "free-news.p.rapidapi.com"        
        }

      responses= requests.request("GET", url, headers=headers,params=qry)
    
      news_details = responses.json()

      news_articles = news_details['articles']

      news_art_lst = []

      for newsart in news_articles:

          title = newsart['title']
          published_date = newsart['published_date']
          summary = newsart['summary']

          topic = newsart['topic']
          source=newsart['clean_url']
          source_link=newsart['link']

          news_id = newsart['_id']

          newsart_dic = {'title':title,'published_date':published_date,'summary':summary,'topic':topic,'source':source,'source_link':source_link,
                    'news_src':'rapidapi','news_id':news_id}

          news_art_lst.append(newsart_dic)

      return news_art_lst

if __name__ == '__main__':

    api_news_reads()
