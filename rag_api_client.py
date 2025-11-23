import httpx
import json
import asyncio
from typing import AsyncGenerator, Tuple

# Note: 'anext' is a built-in function in Python 3.10+, but we'll try to import it for robustness.
try:
    from builtins import anext
except ImportError:
    pass

# Helper function to return an asynchronous generator containing a single error message
async def _async_error_generator(message: str) -> AsyncGenerator[str, None]:
    """Helper to wrap an error message in an async generator."""
    yield message

async def query_rag_api(api_url: str, query: str) -> Tuple[float, AsyncGenerator[str, None]]:
    """
    Sends an asynchronous POST request to the streaming RAG endpoint.
    Retrieves the confidence score from the first chunk, then yields the rest of the text.

    Returns:
        A tuple: (confidence_score, async_generator_of_text_chunks)
    """
    
    # Use httpx.AsyncClient for asynchronous requests
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            # Send the streaming POST request. 
            response = await client.post(
                api_url, 
                params={"query": query}
            )
            
            response.raise_for_status()

        except httpx.ConnectError:
            # Return an async generator instead of a sync one.
            error_msg = "**❌ Connection Error:** Could not connect to the RAG server. Is it running at the correct address?"
            return 0.0, _async_error_generator(error_msg)
        except httpx.HTTPStatusError as e:
            # Return an async generator instead of a sync one.
            error_msg = f"**❌ Server HTTP Error:** Status {e.response.status_code}"
            return 0.0, _async_error_generator(error_msg)
        except Exception as e:
            # Return an async generator instead of a sync one.
            error_msg = f"**❌ An unexpected error occurred:** {str(e)}"
            return 0.0, _async_error_generator(error_msg)

        # --- Stream Processing: Read Metadata ---

        confidence = 0.0
        remaining_text = "" # Initialize variable to hold text that came with the JSON
        
        # 1. Get the async iterator for the bytes stream
        byte_iterator = response.aiter_bytes()
        
        try:
            # 2. Manually advance the async iterator to get the first chunk (metadata)
            metadata_chunk = await anext(byte_iterator, None)
            
            if metadata_chunk:
                # Decode and strip the JSON metadata string
                metadata_str = metadata_chunk.decode('utf-8').strip()
                
                # Robustness fix: Find the last '}' which marks the end of the JSON object,
                # in case the server sent the first text token immediately after.
                try:
                    # Find the index of the last closing brace and include it (+1)
                    json_end = metadata_str.rindex('}') + 1
                    json_part = metadata_str[:json_end]
                    
                    # Store any remaining text data that belongs to the answer
                    remaining_text = metadata_str[json_end:].strip()
                except ValueError:
                    # If '}' is not found, the chunk is corrupted or incomplete JSON
                    # Raise a JSONDecodeError to be caught below
                    raise json.JSONDecodeError("Missing closing brace for JSON metadata.", metadata_str, 0)
                    
                
                metadata = json.loads(json_part)
                confidence = metadata.get("confidence", 0.0)
            
        except StopAsyncIteration:
            await response.aclose()
            # Return an async generator instead of a sync one.
            error_msg = "**❌ Stream Error:** Server returned an empty response before metadata."
            return 0.0, _async_error_generator(error_msg)
        except json.JSONDecodeError as e:
            await response.aclose()
            # Return an async generator instead of a sync one.
            error_msg = f"**❌ Parsing Error:** Failed to decode JSON metadata from server. Raw part: {json_part if 'json_part' in locals() else metadata_str}. Error: {e}"
            return 0.0, _async_error_generator(error_msg)

        # --- Text Streaming: Create the Async Generator ---
        
        async def text_streamer() -> AsyncGenerator[str, None]:
            """Async generator that yields the initial remaining text and the subsequent chunks."""
            
            # If the initial chunk contained text after the JSON, yield it first
            if remaining_text:
                yield remaining_text
                
            # Then yield the rest of the stream
            async for chunk in byte_iterator:
                yield chunk.decode('utf-8')
            
            # IMPORTANT: Close the response when the generator is exhausted
            await response.aclose()


        # Return the extracted confidence and the async generator object
        return confidence, text_streamer()