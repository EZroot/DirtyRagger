import requests
from bs4 import BeautifulSoup, Comment
from typing import List, Dict

class WebScraper:
    """
    Utility class for performing web searches and scraping clean text from URLs.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        # DuckDuckGo HTML search endpoint
        self.search_url = "https://duckduckgo.com/html"
        print("WebScraper initialized.")

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        print(f"\n[Search] Looking up: '{query}'...")

        url = "https://lite.duckduckgo.com/lite/"
        params = {"q": query}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[Search Error] {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        # Results contain <a class="result-link" href="http://...">
        for a in soup.find_all("a", {"class": "result-link"}, limit=num_results):
            title = a.get_text(strip=True)
            url = a.get("href")
            if url and url.startswith("http"):
                results.append({"title": title, "url": url})

        print(results)
        return results



    def scrape_url(self, url: str) -> str:
        print(f"[Scrape] Cleaning text from: {url}")
        
        try:
            content_response = requests.get(url, headers=self.headers, timeout=10)
            content_response.raise_for_status()
            
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            
            for tag in content_soup(["script", "style"]):
                tag.decompose()

            for tag in content_soup.find_all(['nav', 'header', 'footer', 'aside', 'form']):
                tag.decompose()
                
            body_content = content_soup.find('body')
            if body_content:
                plain_text = body_content.get_text(separator=' ', strip=True)
                return plain_text
            
            return "Could not find body tag for full cleanup."
                
        except requests.exceptions.RequestException as e:
            return f"Error fetching content for cleanup: {e}"


if __name__ == "__main__":
    scraper = WebScraper()
    search_query = "osrs kreeara strat"

    results = scraper.search(search_query, num_results=3)

    if results:
        print("\n--- Extracted Search Results (Title and URL) ---")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Title: {result['title']}")
            print(f"  URL:   {result['url']}")
            print("-" * 20)

        first_url = results[0]['url']
        plain_text = scraper.scrape_url(first_url)

        print(f"\nExtracted Plain Text Snippet ({len(plain_text)} chars):")
        print(plain_text[:500] + ('...' if len(plain_text) > 500 else ''))
    else:
        print("\nNo search results could be extracted.")
