import re
import json
from web_scraper import WebScraper 

MAX_SCRAPE_CHARS = 2000

class AvailableTools:
    def __init__(self):
        self.webscraper = WebScraper()

    def web_search(self, query: str, max_scraped_chars: int = MAX_SCRAPE_CHARS) -> str:
        results = self.webscraper.search(query, num_results=3)
        
        if results:
            print("\n--- Extracted Search Results (Title and URL) ---")
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Title: {result['title']}")
                print(f"  URL:   {result['url']}")
                print("-" * 20)
        else:
            combined_query = f"Context:\nNo results found for the query."
            return combined_query
        
        plain_text = self.webscraper.scrape_url(results[0]['url'])
        combined_query = f"Context:\n{plain_text[:max_scraped_chars]}"
        return combined_query

class ToolParseAndExecute:
    def __init__(self):
        self.tools = AvailableTools()

    def parse_tool_call(self, llm_output: str) -> dict:
        TOOL_PATTERN = r"<tool_call>\s*(.*?)\s*</tool_call>"
        match = re.search(TOOL_PATTERN, llm_output, re.DOTALL)

        tool_dict = None

        if match:
            json_str = match.group(1).strip()
            
            print(f"--- Captured JSON String ---")
            print(json_str)
            
            try:
                # Parse the JSON string into a Python dictionary
                tool_dict = json.loads(json_str)
                
                print("\n--- Parsed Python Dictionary ---")
                print(tool_dict)
                print(f"\nTool Name: {tool_dict.get('name')}")
                tool_name = tool_dict.get('name')

                # use built in hardcoded tools first
                if tool_name == "web_search":
                    query = tool_dict['arguments'].get('query', '')
                    result = self.tools.web_search(query)
                    return f"**YOU ALREADY SEARCHED THE WEB AND GOT THIS RESULT:{result}**"
                
                return None
            except json.JSONDecodeError as e:
                print(f"\nERROR: Failed to decode JSON content: {e}")
                return None
        else:
            print("ERROR: No <tool_call> tags found in the output.")
            return None