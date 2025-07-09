import asyncio
import os
from dotenv import load_dotenv
import json
from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.agents import ChatCompletionAgent

def create_kernel_with_chat_completion(model_id: str, model_provider: str, service_id:str) -> Kernel:
    kernel = Kernel()

    if model_provider == "OpenAI":
        client = AsyncOpenAI(
            api_key=os.environ["GITHUB_TOKEN"], 
            base_url="https://models.inference.ai.azure.com/",
        )

        kernel.add_service(
            OpenAIChatCompletion(
                ai_model_id=model_id,
                async_client=client,
                service_id=service_id
            )
        )
    elif model_provider == "DeepSeek":
        client = AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], 
            base_url="https://api.deepseek.com/v1"
        )

        kernel.add_service(
            OpenAIChatCompletion(
                ai_model_id=model_id,
                async_client=client,
                service_id=service_id
            )
        )
    elif model_provider == "Gemini":
        api_key=os.environ["GOOGLE_API_KEY"]
        chat_completion_service = GoogleAIChatCompletion(
            gemini_model_id=model_id,
            api_key=api_key,
            service_id=service_id
        )
        kernel.add_service(chat_completion_service)

    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()    
    return kernel, settings

def create_agent(name: str, kernel: Kernel, instructions: str, plugins: list) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        name= name,
        instructions=instructions,
        kernel=kernel,
        plugins=plugins
    )
