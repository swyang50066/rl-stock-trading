import  os
import  sys
from    datetime            import  datetime, timedelta

import  numpy               as      np
import  pandas              as      pd
import  pandas_datareader   as      pdr

import  urllib.request
from    urllib              import  parse
from    urllib.parse        import  quote
from    bs4                 import  BeautifulSoup

from    konlpy.tag          import  Okt
from    collections         import  Counter

num_article = 0

def AppleBananaCherry():
    # Parameters 
    output_file_name = 'test.txt' 
    keyword  = ''
    date_str = '2019.01.01'
    date_end = '2020.01.20'

    stock_info = getStockCode('kospi')
    dates      = np.arange(datetime(2019,1,1).date(), datetime(2020,1,20).date(), timedelta(days=1))
    signal_map = pd.DataFrame(np.zeros((len(dates), len(stock_info['company']))), 
                              index=dates, 
                              columns=stock_info['company'])

    counts = [0 ,0]
    for page in range(1000) :#range(267):
        page_num = str(15*page + 1)
        url = ('https://m.search.naver.com/search.naver?' +
               'where=m_news' + 
               '&query=' + quote('임상') + '+' + quote('성공') + '+' + quote('소식') +
               '&oquery=' + quote('임상') +'%25' + quote('성공') + '%25' + quote('소식')+ 
               '&sm=tab_opt' +
               '&sort=0'  +
               '&photo=0' +
               '&field=0' +
               '&reporter_article=' +
               '&pd=3' +
               '&ds=' + date_str +
               '&de=' + date_end +
               '&start=' + page_num)

        print(url)

        # Get article text from URL
        article      = read_article(url)
        article_url  = article.get_url()
        signal_map, counts = article.get_text(article_url, stock_info, signal_map, counts)
    
        print('here', np.sum(signal_map.values))
        print(counts)

class read_article(object):
    def __init__(self, url):
        self.url = url
    
    def get_url(self):
        # Request html source from the URL
        url_source = urllib.request.urlopen(self.url)

        # Extract article text from body html of url_source (To encode Hangul, encoding method is set 'utf-8')
        soup = BeautifulSoup(url_source, 'html.parser', from_encoding='utf-8')
        tags = soup.find_all('div', class_='news_dsc')
        
        sub_urls = []
        for tag in tags:
            sub_urls.append( tag.select('a')[0]['href'])
        
        return sub_urls 
    
    def get_text(self, sub_urls, stock_info, signal_map, counts):
        date, text = '', ''
        for i, sub_url in enumerate(sub_urls):
            print('Processing: %d/%d' % (i+1, 150))
            # Request html source from the URL
            url_source = urllib.request.urlopen(sub_url)

            # Extract article text from body html of url_source (To encode Hangul, encoding method is set 'utf-8') 
            soup = BeautifulSoup(url_source, 'html.parser', from_encoding='utf-8')
            date_tags    = soup.find_all('div', class_='media_end_head_info_datestamp_bunch')
            article_tags = soup.find_all('div', class_='newsct_article _news_article_body')
				
            for date_tag in date_tags: 
                date = str(date_tag.select('span')[0])
                date = date.split('>')[1].split()[0].split('.')[:-1]
                date = '-'.join(date)

            for article_tag in article_tags:
                #text = self.remove_symbol(str(article_tag.find_all(text=True)) + 'GH신소재').split()
                text = self.remove_symbol(str(article_tag.find_all(text=True))).split()
          
            signals = np.where(np.isin(stock_info['company'], text).astype(np.int) > 0)[0]
            if len(signals) == 0: continue
            
            for signal in signals:
                stock_name = signal_map.columns[signal]
                stock_code = stock_info.loc[stock_info['company'] == stock_name, 'code'].values[0]
                stock_data = pdr.DataReader('005930.KS', 'yahoo', datetime(2019, 1, 1), datetime(2020, 1, 20))
                #stock_data = pdr.get_data_yahoo(str(stock_code) + '.KQ') # or '.KQ'

                count = 0; skip = False
                while date not in stock_data.index.strftime('%Y-%m-%d'):
                    date = str(datetime.strptime(date, '%Y-%m-%d').date() + timedelta(days=1))
                    count += 1
                    if count == 5: skip = True; break
                
                if skip == True: continue

                date_next = date
                for _ in range(3):                
                    date_next = str(datetime.strptime(date_next, '%Y-%m-%d').date() + timedelta(days=1))
                    count = 0; skip = False

                    while date_next not in stock_data.index.strftime('%Y-%m-%d'):
                        date_next = str(datetime.strptime(date_next, '%Y-%m-%d').date() + timedelta(days=1))
                        count += 1
                        if count == 5: skip = True; break
                
                    if skip == True: continue

                    value = stock_data['Open'].loc[stock_data['Open'].index == date].values
                    value_next = stock_data['High'].loc[stock_data['High'].index == date_next].values
                    delta_value = ((value_next - value)/value)[0]
                    #print(stock_name, delta_value)
                
                    if delta_value > 0.05: 
                        signal_map[stock_name].loc[signal_map[stock_name].index == date] += 1
                        counts[0] += 1
                        break

                    elif delta_value < 0: 
                        signal_map[stock_name].loc[signal_map[stock_name].index == date] -= 1
                        counts[1] += 1
                    

        return signal_map, counts
    
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

def getStockCode(market):
    market = 'kosdaq'
    if market == 'kosdaq':
        url_market = 'kosdaqMkt'
    elif market == 'kospi':
        url_market = 'stockMkt'
    
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType=%s' % url_market
    stock_info = pd.read_html(url, header=0)[0][['회사명', '종목코드']]
    stock_info.columns = ['company', 'code']

    return stock_info

if __name__=='__main__':
    AppleBananaCherry() 
