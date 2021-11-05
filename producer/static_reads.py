import json

def static_news_reads():

     news_art_lst = []

     with open('dataset/news_data.json', mode='r', errors='ignore') as jsonf:       
            
       for dict in jsonf:
           news_art_lst.append(json.loads(dict))
          
       return news_art_lst

if __name__ == '__main__':

    static_news_reads()
