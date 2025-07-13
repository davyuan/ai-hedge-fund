import json
import asyncio
import os
from typing import Annotated
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from semantic_kernel.functions import kernel_function
from src.mcp.client import mcp_read_state, mcp_upsert_state

class PorfolioDataPlugin:

    @kernel_function(description="Provides portfolio data for analysis for the Portfolio Manager based on the stock ticker.")
    async def get_porforlio_data(self, ticker: Annotated[str, "Stock Ticker"]) -> Annotated[str, "Returns portfolio data based on the stock ticker"]:
        #print(f"get_porforlio_data is called with {ticker}")
        result = await mcp_read_state()
        state = result[0].text
        state = json.loads(state)

        return json.dumps(state["data"]["portfolio"])
