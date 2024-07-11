import unittest
from src.data.fetch_news import fetch_news

class TestFetchNews(unittest.TestCase):
    def test_fetch_news(self):
        api_key = 'your_news_api_key'
        articles = fetch_news(api_key, 'AI', '2024-01-01', '2024-01-02')
        self.assertGreater(len(articles), 0)

if __name__ == '__main__':
    unittest.main()

