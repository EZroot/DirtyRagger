import requests
import json
from time import sleep

# --- Configuration ---
BASE_URL = "http://0.0.0.0:8000"
ENDPOINT = "/query"
API_URL = f"{BASE_URL}{ENDPOINT}"

# The query we want to send, this will be properly URL-encoded by requests
EXAMPLE_QUERY = "What is data-oriented design in C#?"

def query_streaming_api(api_url: str, query: str):
    """
    Sends a POST request to the streaming endpoint and processes the response.
    
    The server streams a multi-part response:
    1. A single JSON object for metadata (e.g., confidence score).
    2. A continuous text stream for the main answer.
    """
    
    print(f"-> Querying endpoint: {api_url}")
    print(f"-> With query: '{query}'\n")

    # Use the 'params' argument for query string parameters and 'stream=True' 
    # to handle the response body chunk by chunk.
    try:
        response = requests.post(
            api_url, 
            params={"query": query}, 
            stream=True,
            timeout=10 # Set a reasonable timeout
        )
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return

    # --- Stream Processing ---
    
    # 1. Read and process the first chunk (expected to be JSON metadata)
    try:
        # Use iter_content with a small chunk size to reliably capture the first piece
        iterator = response.iter_content(chunk_size=1024)
        
        # Get the first chunk
        first_chunk = next(iterator, None)
        
        if not first_chunk:
            print("❌ Error: Server returned an empty response.")
            return

        # Attempt to decode the first chunk as JSON
        metadata = json.loads(first_chunk.decode('utf-8'))
        
        print("--- Metadata Received ---")
        # Print the metadata in a readable format
        print(json.dumps(metadata, indent=4))
        print("-------------------------\n")
        
    except (StopIteration, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"❌ Error decoding initial JSON metadata: {e}")
        print("The server might not have started streaming yet or the format is incorrect.")
        return

    # 2. Read and process the remaining chunks (expected to be text stream)
    print("--- Streaming Answer ---")
    
    # Process the rest of the stream
    for chunk in iterator:
        # Check if we received any data
        if chunk:
            # Decode the chunk and print it immediately without a newline
            print(chunk.decode('utf-8'), end="", flush=True)

    print("\n\n--- Stream Finished ---")
    
if __name__ == "__main__":
    query_streaming_api(API_URL, EXAMPLE_QUERY)