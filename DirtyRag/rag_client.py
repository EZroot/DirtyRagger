import requests
import json

# --- Configuration ---
BASE_URL = "http://0.0.0.0:8000"
ENDPOINT = "/query"
ENDPOINT_WEB = "/websearch_query"
API_URL = f"{BASE_URL}{ENDPOINT}"

def query_api_non_streaming(api_url: str, query: str):
    """
    Sends a POST request to the API and processes the single JSON response.
    """
    
    print("WEBSEARCH? (y/n): ", end="")
    use_websearch = input().strip().lower() == 'y'
    if use_websearch:
        api_url = f"{BASE_URL}{ENDPOINT_WEB}"
    else:
        api_url = f"{BASE_URL}{ENDPOINT}"
        
    print(f"\n-> Querying endpoint: {api_url}")
    print(f"-> With query: '{query}'")

    try:
        # 1. Send POST request WITHOUT stream=True
        # Use 'params' for the query string parameter
        response = requests.post(
            api_url, 
            params={"query": query}, 
            # stream=True is removed
            timeout=60 
        )
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        # Catch connection errors, DNS failures, and HTTP error responses
        print(f"❌ Connection/HTTP Error: {e}")
        return

    # --- Response Processing ---
    
    try:
        # 2. FIX: Read the entire response body as a single JSON object
        response_data = response.json()
        
        # 3. FIX: Extract the relevant 'response' text from the JSON object
        generated_text = response_data.get("response", "No response text found.")

        print("\nQwen AI:")
        # Print the final result
        print(generated_text)
        
    except json.JSONDecodeError:
        print(f"❌ Error: Server did not return a valid JSON object. Received status code {response.status_code}.")
        print(f"Raw response text: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


# =========================================================================
# Main execution block
# =========================================================================
if __name__ == "__main__":
    print("Welcome to the Non-Streaming API Client. Type 'quit' or 'exit' to stop.")
    while True:
        # Prompt the user for input
        q = input("Query> ").strip()
        
        # Check for exit commands
        if q.lower() in ('quit', 'exit'):
            print("Exiting client.")
            break
            
        # Ignore empty queries
        if not q:
            continue
            
        # Call the function to query the API with the user's input
        query_api_non_streaming(API_URL, q)