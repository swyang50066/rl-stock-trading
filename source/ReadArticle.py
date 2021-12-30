import  os
import  sys
import  time

import  numpy               as  np
import  pandas              as  pd

import  urllib.request
from    urllib.parse        import  quote
from    bs4                 import  BeautifulSoup

from    konlpy.tag          import  Okt
from    collections         import  Counter

def main():
    # Parameters   
    output_file_name = 'output.txt'
    url = ('http://find.mk.co.kr/new/search.php?pageNum=1&cat=&cat1=&media_eco=&pageSize=20&sub=news&dispFlag=OFF&page=news&s_kwd=' 
            + quote('고기') 
            + '&s_page=total&go_page=&ord=1&ord1=1&ord2=0&s_keyword=' 
            + quote('고기')
            + '&y1=1991&m1=01&d1=01&y2=2019&m2=12&d2=01&area=ttbd')

    # Open file which saves article text
    with open(output_file_name, 'w') as output_file:
        # Get article text from URL
        article = read_article(url)
        article_url  = article.get_url()
        article_text = article.get_text(article_url)
        article_text = article.remove_symbol(article_text)
        word_count   = article.get_tags(article_text, 10) 
        
        df = pd.DataFrame(word_count, index=np.linspace(1,10,10,dtype=np.uint8))
        print(df)

        for pair in word_count:
            noun = pair['tag']
            count = pair['count']
            
            # Write words in articles
            output_file.write('{} {}\n'.format(noun, count))

class read_article(object):
    def __init__(self, url):
        self.url = url
    
    def get_url(self):
        # Request html source from the URL
        url_source = urllib.request.urlopen(self.url)

        # Extract article text from body html of url_source (To encode Hangul, encoding method is set 'utf-8')
        soup = BeautifulSoup(url_source, 'html.parser', from_encoding='utf-8')
        tags = soup.find_all('div', class_='sub_list')
        url_list = {}
        for i, tag in enumerate(tags):
            sub_title = tag.select('a')
            sub_url   = sub_title[0]['href']
            url_list[i] = sub_url
        
        return url_list 
    
    def get_text(self, url_list):
        text = ''
        for key in url_list.keys():
            # Request html source from the URL
            url_source = urllib.request.urlopen(url_list[key])
    
            # Extract article text from body html of url_source (To encode Hangul, encoding method is set 'utf-8') 
            soup = BeautifulSoup(url_source, 'html.parser', from_encoding='utf-8')
            tags = soup.find_all('div', class_='art_txt')
            
            for tag in tags:
                text += str(tag.find_all(text=True))
        
        return text
    
    def remove_symbol(self, text):
        symbols = ["\\n", "\\", "/", "|",
                   "'", '"', ",", ".", "!", "?", 
                   "(", ")", "{", "}", "[", "]", "<", ">",
                   "@", "#", "$", "%", "^", "&", "*", 
                   "+", "-", "_", "=", "~", 
                   "ⓒ", "▶"]
        for s in symbols:
            text = text.replace(s, "")
        return text

    def get_tags(self, text, ntags=50):
        spliter = Okt()
        nouns = spliter.nouns(text)
        count = Counter(nouns)
        return_list = []
    
        for n, c in count.most_common(ntags):
            temp = {'tag': n, 'count': c}
            return_list.append(temp)
    
        return return_list

if __name__=='__main__':
    main() 
