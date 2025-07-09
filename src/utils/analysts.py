"""Constants and utilities related to analysts configuration."""

import json

AGENT_CONFIG_FILE_PATH = "agents_config.json" 
ANALYST_ORDER= None

try:
    with open(AGENT_CONFIG_FILE_PATH, 'r') as f:
        ANALYST_CONFIG = json.load(f)
        ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), 
            key=lambda x: x[1]["order"])
            if config["order"] <= 10
        ]

    print(f"Successfully loaded '{AGENT_CONFIG_FILE_PATH}' into ANALYST_CONFIG.")
except FileNotFoundError:
    print(f"Error: The file '{AGENT_CONFIG_FILE_PATH}' was not found.")
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from '{AGENT_CONFIG_FILE_PATH}'. Check file format.\n{e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, instructions, user_message_template, plugins) tuples."""
    nodes = {}
    for key, config in ANALYST_CONFIG.items():
        plugin_names = config["agent_plugins"]
        nodes[key] = (f"{key}_agent", config["agent_instructions"], config["agent_message_template"], plugin_names)

    return nodes

def get_agents_list():
    """Get the list of agents for API responses."""
    return [
        {
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "investing_style": config["investing_style"],
            "order": config["order"]
        }
        for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    ]


def get_investing_styles():
    """Get all unique investing styles."""
    return list(set(config["investing_style"] for config in ANALYST_CONFIG.values()))


def get_investing_style_display_names():
    """Get display names for investing styles."""
    return {
        "value_investing": "Value Investing",
        "growth_investing": "Growth Investing", 
        "contrarian_activist": "Contrarian/Activist",
        "macro_global": "Macro/Global",
        "technical_analysis": "Technical Analysis",
        "quantitative_analytical": "Quantitative/Analytical"
    }


def get_agents_by_investing_style():
    """Get agents grouped by investing style."""
    groups = {}
    for key, config in ANALYST_CONFIG.items():
        style = config["investing_style"]
        if style not in groups:
            groups[style] = []
        groups[style].append({
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "order": config["order"]
        })
    
    # Sort agents within each group by order
    for style in groups:
        groups[style].sort(key=lambda x: x["order"])
    
    return groups
