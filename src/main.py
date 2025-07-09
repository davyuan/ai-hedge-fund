import sys
import asyncio
import json
import re
import argparse
from datetime import datetime

from dotenv import load_dotenv
from colorama import Fore, Style, init
import questionary
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
from src.utils.ollama import ensure_ollama_and_model
from src.utils.agents import create_kernel_with_chat_completion, create_agent
from src.mcp.client import mcp_read_state, mcp_upsert_state
from src.plugins.cathie_wood import AnalysisDataPlugin4CathieWood
from src.plugins.aswath_damodaran import AnalysisDataPlugin4AswathDamodaran
from src.plugins.ben_graham import AnalysisDataPlugin4BenGraham
from src.plugins.bill_ackman import AnalysisDataPlugin4BillAckman
from src.plugins.charlie_munger import AnalysisDataPlugin4CharlieMunger
from src.plugins.michael_burry import AnalysisDataPlugin4MichaelBurry
from src.plugins.peter_lynch import AnalysisDataPlugin4PeterLynch
from src.plugins.phil_fisher import AnalysisDataPlugin4PhilFisher
from src.plugins.rakesh_jhunjhunwala import AnalysisDataPlugin4RakeshJhunjhunwala
from src.plugins.stanley_druckenmiller import AnalysisDataPlugin4StanleyDruckenmiller
from src.plugins.warren_buffett import AnalysisDataPlugin4WarrenBuffett
from src.plugins.portfolio_manager import PorfolioDataPlugin
from src.plugins.risk_manager import RiskDataPlugin
from src.utils.visualize import save_graph_as_png
from dateutil.relativedelta import relativedelta
from semantic_kernel.agents import ChatCompletionAgent

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)

def parse_hedge_fund_response(response: str):
    """Parses a JSON string and returns a dictionary."""
    try:
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)

        if match:
            json_string = match.group(1).strip() # Extract the content and strip whitespace
            parsed_data = json.loads(json_string)
            return parsed_data
        else:
            parsed_data = json.loads(response)
            return parsed_data
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


async def invoke_agent(agent: ChatCompletionAgent, user_message:str):
    async for response in agent.invoke(
        messages = user_message
    ):
        return response

##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o-mini",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    state =  {
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            }
    result = asyncio.run(mcp_upsert_state(state))

    # Create the agents with seclected analysts
    agents = create_agents(selected_analysts=selected_analysts, model_id=model_name, model_provider=model_provider)

    try:
        for ticker in tickers:
            response = None
            analysis_data = {}
            for agent, user_message_template in agents:
                # Format the user message using the template and arguments
                user_message = user_message_template.replace("{$ticker}", ticker).replace("{$end_date}", end_date)
                if response != None :
                    user_message = user_message.replace("{$analysis_data}", str(response.content))
                #print(f"user_message:{user_message}")

                response = asyncio.run(invoke_agent(agent, user_message))
                if agent.name != "portfolio_manager_agent":
                    analysis_data[f"{agent.name}"] = parse_hedge_fund_response(str(response.content))

        return {
            "decisions": parse_hedge_fund_response(str(response.content)),
            "analyst_signals": analysis_data
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state

def create_agents(selected_analysts=None, model_id="gpt-4o-mini", model_provider="OpenAI", service_id="HedgeFundAgent"):
    agents = []

    kernel, settings = create_kernel_with_chat_completion(model_id=model_id, model_provider=model_provider, service_id=service_id)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()
    # Add selected analyst nodes
    plugin_registry = {
        "AnalysisDataPlugin4AswathDamodaran": AnalysisDataPlugin4AswathDamodaran,
        "AnalysisDataPlugin4CathieWood": AnalysisDataPlugin4CathieWood,
        "AnalysisDataPlugin4BenGraham": AnalysisDataPlugin4BenGraham,
        "AnalysisDataPlugin4BillAckman": AnalysisDataPlugin4BillAckman,
        "AnalysisDataPlugin4CharlieMunger": AnalysisDataPlugin4CharlieMunger,
        "AnalysisDataPlugin4MichaelBurry": AnalysisDataPlugin4MichaelBurry,
        "AnalysisDataPlugin4PeterLynch": AnalysisDataPlugin4PeterLynch,
        "AnalysisDataPlugin4PhilFisher": AnalysisDataPlugin4PhilFisher,
        "AnalysisDataPlugin4RakeshJhunjhunwala": AnalysisDataPlugin4RakeshJhunjhunwala,
        "AnalysisDataPlugin4StanleyDruckenmiller": AnalysisDataPlugin4StanleyDruckenmiller,
        "AnalysisDataPlugin4WarrenBuffett": AnalysisDataPlugin4WarrenBuffett,
        "PorfolioDataPlugin": PorfolioDataPlugin,
        "RiskDataPlugin": RiskDataPlugin
        }

    for analyst_key in selected_analysts:
        node_name, instructions, user_message_template, plugins = analyst_nodes[analyst_key]
        plugin_instances = []
        for plugin_name in plugins:
            if plugin_name in plugin_registry:
                instance = plugin_registry[plugin_name]()
                plugin_instances.append(instance)

        agent = create_agent(name=node_name, kernel= kernel, instructions=instructions, plugins=plugin_instances)
        agents.append((agent, user_message_template))

    #ad the portfolio manager agent to the chain of agents
    node_name, instructions, user_message_template, plugins = analyst_nodes["portfolio_manager"]
    plugin_instances = []
    for plugin_name in plugins:
        if plugin_name in plugin_registry:
            instance = plugin_registry[plugin_name]()
            plugin_instances.append(instance)

    agent = create_agent(name=node_name, kernel= kernel, instructions=instructions, plugins=plugin_instances)
    agents.append((agent, user_message_template))

    return agents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash position. Defaults to 100000.0)")
    parser.add_argument("--margin-requirement", type=float, default=0.0, help="Initial margin requirement. Defaults to 0.0")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of stock ticker symbols")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Select analysts
    selected_analysts = None
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

    # Select LLM model based on whether Ollama is being used
    model_name = ""
    model_provider = ""

    if args.ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")

        # Select from Ollama-specific models
        model_name: str = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_name:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        if model_name == "-":
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

        # Ensure Ollama is installed, running, and the model is available
        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            sys.exit(1)

        model_provider = ModelProvider.OLLAMA.value
        print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    else:
        # Use the standard cloud-based LLM selection
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        model_name, model_provider = model_choice

        # Get model info using the helper function
        model_info = get_model_info(model_name, model_provider)
        if model_info:
            if model_info.is_custom():
                model_name = questionary.text("Enter the custom model name:").ask()
                if not model_name:
                    print("\n\nInterrupt received. Exiting...")
                    sys.exit(0)

            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": args.initial_cash,  # Initial cash amount
        "margin_requirement": args.margin_requirement,  # Initial margin requirement
        "margin_used": 0.0,  # total margin usage across all short positions
        "positions": {
            ticker: {
                "long": 0,  # Number of shares held long
                "short": 0,  # Number of shares held short
                "long_cost_basis": 0.0,  # Average cost basis for long positions
                "short_cost_basis": 0.0,  # Average price at which shares were sold short
                "short_margin_used": 0.0,  # Dollars of margin used for this ticker's short
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # Realized gains from long positions
                "short": 0.0,  # Realized gains from short positions
            }
            for ticker in tickers
        },
    }

    # Run the hedge fund
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )
    print_trading_output(result)
