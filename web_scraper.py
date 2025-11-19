import requests
from bs4 import BeautifulSoup, Comment
from typing import List, Dict

class WebScraper:
    """
    Utility class for performing web searches and scraping clean text from URLs.
    """
    def __init__(self):
        # Default headers to mimic a common browser, helping to avoid immediate blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.search_url = "https://duckduckgo.com/html"
        print("WebScraper initialized.")

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Performs a web search and returns a list of dictionaries with titles and URLs.
        
        Args:
            query (str): The search query.
            num_results (int): Maximum number of search results to return.
            
        Returns:
            List[Dict]: A list of results, e.g., [{'title': '...', 'url': '...'}]
        """
        params = {'q': query}
        
        print(f"\n[Search] Looking up: '{query}'...")
        
        try:
            # 1. Make the HTTP Request
            response = requests.get(self.search_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status() 
            
        except requests.exceptions.RequestException as e:
            print(f"[Search Error] Failed to fetch search results: {e}")
            return []

        # 2. Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # HTML Cleanup: Phase 1 (Removing unwanted tags from search result page)
        for tag in soup(["script", "style"]):
            tag.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 3. Extract Structured Search Results
        results = []
        result_containers = soup.find_all('div', class_='result', limit=num_results)

        for container in result_containers:
            title_tag = container.find('a', class_='result__a')
            link_tag = container.find('a', class_='result__url')

            if title_tag and link_tag:
                title = title_tag.text.strip()
                url = link_tag['href']
                
                results.append({
                    'title': title,
                    'url': url
                })

        return results

    def scrape_url(self, url: str) -> str:
        """
        Fetches a specific webpage and extracts cleaned, plain text content.
        
        Args:
            url (str): The URL of the page to scrape.
            
        Returns:
            str: The cleaned plain text content of the page.
        """
        print(f"[Scrape] Cleaning text from: {url}")
        
        try:
            # Fetch the actual content page
            content_response = requests.get(url, headers=self.headers, timeout=10)
            content_response.raise_for_status()
            
            content_soup = BeautifulSoup(content_response.text, 'html.parser')
            
            # HTML Cleanup: Phase 2 (Removing typical non-content elements)
            # Remove scripts, styles, and comments
            for tag in content_soup(["script", "style"]):
                tag.decompose()

            # Remove common non-content tags (navigation, headers, footers, sidebars)
            for tag in content_soup.find_all(['nav', 'header', 'footer', 'aside', 'form']):
                tag.decompose()
                
            # Get plain text from the remaining body content
            body_content = content_soup.find('body')
            if body_content:
                # Get text, strip excess whitespace, and normalize line breaks
                plain_text = body_content.get_text(separator=' ', strip=True)
                return plain_text
            
            return "Could not find body tag for full cleanup."
                
        except requests.exceptions.RequestException as e:
            return f"Error fetching content for cleanup: {e}"


if __name__ == "__main__":
    # Example usage:
    scraper = WebScraper()
    search_query = "osrs kreeara strat" 
    
    # Step 1: Perform the search and extract structured results
    extracted_results = scraper.search(search_query, num_results=3)

    if extracted_results:
        print("\n--- Extracted Search Results (Title and URL) ---")
        for i, result in enumerate(extracted_results):
            print(f"Result {i+1}:")
            print(f"  Title: {result['title']}")
            print(f"  URL:   {result['url']}")
            print("-" * 20)
        
        # Step 2: Demonstrate full content cleanup on the first result
        first_url = extracted_results[0]['url']
        plain_text = scraper.scrape_url(first_url)
        
        print(f"\nExtracted Plain Text Snippet ({len(plain_text)} chars):")
        # Print only the first 500 characters
        print(plain_text[:500] + ('...' if len(plain_text) > 500 else ''))
    else:
        print("\nNo search results could be extracted.")