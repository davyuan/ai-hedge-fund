# server.py
from mcp.server.fastmcp import FastMCP
from typing_extensions import Annotated, Sequence, TypedDict
import json
from src.graph.state import AgentState
import logging
import sys
import os

LOG_FILE_PATH = "mcp_server_debug.log" # Define your log file path
STATE_FILE_PATH = "agent_state.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG to see all debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH)
        #logging.StreamHandler(sys.stderr)
    ]
)

# Get a logger for your module/server
logger = logging.getLogger(__name__)
logger.info("MCP Server started, sending logs to log file.")

# Create an MCP server
mcp = FastMCP("AgentState")

def load_state_from_file() -> AgentState | None:
    """Loads the AgentState from the JSON file."""
    if not os.path.exists(STATE_FILE_PATH):
        logger.info(f"State file not found at {STATE_FILE_PATH}. Returning initial None state.")
        return None # Or return a default empty state like {"messages": [], "data": {}, "metadata": {}}

    try:
        with open(STATE_FILE_PATH, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and all(k in data for k in ["data", "metadata"]):
                logger.info(f"State successfully loaded from {STATE_FILE_PATH}.")
                return data
            else:
                logger.warning(f"Data in {STATE_FILE_PATH} is not a valid AgentState structure. Returning None.")
                return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from state file {STATE_FILE_PATH}: {e}. Returning None.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading state from file {STATE_FILE_PATH}: {e}. Returning None.", exc_info=True)
        return None

def save_state_to_file(current_state: AgentState | None):
    """Saves the current AgentState to the JSON file."""
    try:
        # Ensure the directory exists, although in this case, it's server.py's dir
        with open(STATE_FILE_PATH, 'w') as f:
            # json.dump can handle None, which will write 'null' to the file
            json.dump(current_state, f, indent=4) # indent for readability
        logger.info(f"State successfully saved to {STATE_FILE_PATH}.")
    except TypeError as e:
        logger.error(f"TypeError: State object is not JSON serializable: {e}. State: {current_state}", exc_info=True)
    except Exception as e:
        logger.error(f"Error saving state to file {STATE_FILE_PATH}: {e}.", exc_info=True)

#MCP tool functions
@mcp.tool()
def get() -> AgentState | None:
    state = load_state_from_file()
    return state

@mcp.tool()
def set(new_state: AgentState | None):
    """Updates the in-memory global state and saves it to a file."""
    global state
    logger.debug(f"SET tool called. Raw new_state received: {new_state}")

    state = new_state

    try:
        logger.info(f"Global 'state' in memory updated to: {json.dumps(state)}")
        save_state_to_file(state)
    except Exception as e:
        logger.error(f"Error occurred after state update (logging/saving): {e}", exc_info=True)
