import  os
import  sys

from    urllib              import  parse
from    urllib.parse        import  quote
import  urllib.request

from    bs4                 import  BeautifulSoup   as bs

url = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=101&oid=055&aid=0000790342'
url_source = urllib.request.urlopen(url) # ->>> HTML source

soup = bs(url_source, 'html.parser', from_encoding='utf-8')
html_class = soup.find_all('h3', id='articleTitle')#class_='article_info')
html_class = soup.find_all('div', class_='article_info')

print(type(html_class), len(html_class), html_class)

#print(html_class[0].find_all(text=True))
