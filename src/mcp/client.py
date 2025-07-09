from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from src.graph.state import AgentState

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="mcp",
    args=["run", "src/mcp/server.py"],
    env=None,
)

async def mcp_read_state():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write
        ) as session:
            await session.initialize()

            result = await session.call_tool("get")
            return result.content

async def mcp_upsert_state(state: AgentState):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write
        ) as session:
            await session.initialize()

            result = await session.call_tool("set", arguments={"new_state": state})
            return
