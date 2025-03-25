import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

class FinancialNewsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.sources = {
            'yahoo_finance': 'https://finance.yahoo.com/news/',
            'marketwatch': 'https://www.marketwatch.com/latest-news',
            'cnbc': 'https://www.cnbc.com/finance/'
        }
    
    def fetch_articles(self, source, limit=10):
        """Fetch financial news articles from a specific source"""
        if source not in self.sources:
            raise ValueError(f"Source {source} not supported. Available sources: {list(self.sources.keys())}")
        
        url = self.sources[source]
        articles = []
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Different parsing logic for each source
            if source == 'yahoo_finance':
                article_elements = soup.select('h3 a')
                for i, element in enumerate(article_elements):
                    if i >= limit:
                        break
                    
                    title = element.text.strip()
                    link = 'https://finance.yahoo.com' + element['href'] if element.has_attr('href') else ''
                    
                    if link:
                        article_content = self._fetch_article_content(link)
                        articles.append({
                            'title': title,
                            'url': link,
                            'source': source,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'content': article_content
                        })
            
            elif source == 'marketwatch':
                article_elements = soup.select('h3.article__headline a')
                for i, element in enumerate(article_elements):
                    if i >= limit:
                        break
                    
                    title = element.text.strip()
                    link = element['href'] if element.has_attr('href') else ''
                    
                    if link:
                        article_content = self._fetch_article_content(link)
                        articles.append({
                            'title': title,
                            'url': link,
                            'source': source,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'content': article_content
                        })
            
            elif source == 'cnbc':
                article_elements = soup.select('div.Card-titleContainer a')
                for i, element in enumerate(article_elements):
                    if i >= limit:
                        break
                    
                    title = element.text.strip()
                    link = element['href'] if element.has_attr('href') else ''
                    
                    if link:
                        article_content = self._fetch_article_content(link)
                        articles.append({
                            'title': title,
                            'url': link,
                            'source': source,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'content': article_content
                        })
                        
        except Exception as e:
            print(f"Error fetching articles from {source}: {str(e)}")
        
        return articles
    
    def _fetch_article_content(self, url):
        """Fetch the content of an individual article"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # General content extraction (may need adjustment for each specific site)
            if 'yahoo' in url:
                paragraphs = soup.select('div.caas-body p')
            elif 'marketwatch' in url:
                paragraphs = soup.select('div.article__body p')
            elif 'cnbc' in url:
                paragraphs = soup.select('div.ArticleBody-articleBody p')
            else:
                paragraphs = soup.select('p')
            
            content = ' '.join([p.text.strip() for p in paragraphs])
            return content
            
        except Exception as e:
            print(f"Error fetching article content from {url}: {str(e)}")
            return ""
    
    def get_all_news(self, limit_per_source=5):
        """Get news from all available sources"""
        all_articles = []
        
        for source in self.sources.keys():
            articles = self.fetch_articles(source, limit=limit_per_source)
            all_articles.extend(articles)
        
        return pd.DataFrame(all_articles)