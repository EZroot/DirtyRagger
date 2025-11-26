import httpx
from typing import Dict, Any

# WARNING! 
# THIS TIMEOUT IS SET TO 60. May not be sufficient for large responses or image generation requests

async def query_rag_api_json(api_url: str, query: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                api_url, 
                params={"query": query}
            )
            response.raise_for_status()
            return response.json()

        except httpx.ConnectError as e:
            raise RuntimeError(f"Connection Error: Could not connect to the RAG server. {e}")
        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json().get("detail", e.response.text)
            except:
                error_detail = e.response.text
                
            raise RuntimeError(f"Server HTTP Error: Status {e.response.status_code}. Detail: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error occurred: {str(e)}")