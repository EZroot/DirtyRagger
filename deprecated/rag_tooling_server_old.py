# rag_tooling_server.py
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

# ============================================================
# Tooling Server Configuration
# ============================================================
TOOL_SERVER_HOST = "0.0.0.0"
TOOL_SERVER_PORT = 8001

# --- Logging Configuration ---
# Configure the logger to output detailed information
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for maximum output, INFO for general
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# -----------------------------

app = FastAPI(title="RAG Tooling Execution Server")

# ============================================================
# Tool Definitions & Execution Mappers
# ============================================================

def get_current_time(**kwargs) -> str:
    """Returns the current system time in ISO format."""
    logger.info("🛠️ Executing tool: get_current_time")
    # We ignore kwargs here as the function takes no arguments
    current_time = datetime.datetime.now().isoformat()
    logger.debug(f"⏰ Result from get_current_time: {current_time}")
    return current_time

# Mapping of function names (from LLM call) to actual Python callable objects
# NOTE: All functions must accept **kwargs (even if they don't use them)
# to handle argument passing from the LLM gracefully.
TOOL_MAP = {
    "get_current_time": get_current_time,
}

# Pydantic model for the incoming request body
class ToolExecutionRequest(BaseModel):
    function_name: str
    arguments: Dict[str, Any] = {} # Default to empty dictionary

# ============================================================
# API Endpoint
# ============================================================

@app.post("/execute")
async def execute_tool(request: ToolExecutionRequest):
    """
    Executes the specified function with the provided arguments.
    """
    func_name = request.function_name
    args = request.arguments
    
    logger.info(f"➡️ Received execution request for tool: '{func_name}'")
    logger.debug(f"   Arguments: {args}")

    if func_name not in TOOL_MAP:
        # Log the failure before raising the HTTPException
        available_tools = list(TOOL_MAP.keys())
        logger.warning(f"❌ Rejected request. Unknown tool: '{func_name}'. Available: {available_tools}")
        raise HTTPException(
            status_code=404, 
            detail=f"Unknown tool: '{func_name}'. Available tools: {available_tools}"
        )

    try:
        # Get the Python function object
        func = TOOL_MAP[func_name]
        
        # Execute the function, passing the arguments dictionary (**)
        logger.info(f"🚀 Calling Python function: {func_name} with arguments: {args}")
        result = func(**args)
        
        # Log the successful result
        logger.info(f"✅ Tool '{func_name}' executed successfully.")
        logger.debug(f"   Tool Output: {str(result)}")
        
        # Return the result as a JSON response
        return {
            "function_name": func_name,
            "success": True,
            "output": str(result) # Convert result to string for consistent API output
        }
        
    except Exception as e:
        # Catch any execution errors and return them to the RAG server
        error_message = f"Error executing tool '{func_name}': {type(e).__name__}: {str(e)}"
        logger.error(f"🚨 Execution failed for '{func_name}': {error_message}")
        
        raise HTTPException(
            status_code=500, 
            detail=error_message
        )

# ============================================================
# Server Startup
# ============================================================
if __name__ == "__main__":
    logger.info(f"Starting RAG Tooling Execution Server on {TOOL_SERVER_HOST}:{TOOL_SERVER_PORT}...")
    # Uvicorn will handle its own logging, but the initial message is good.
    uvicorn.run(app, host=TOOL_SERVER_HOST, port=TOOL_SERVER_PORT)