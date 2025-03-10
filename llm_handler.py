import os
import json
import re
import logging
from typing import Optional, Dict, Any

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

logger = logging.getLogger(__name__)

class LLMHandler:
    """
    Handles interactions with language models using Semantic Kernel.
    """

    def __init__(self):
        """Initialize the LLM handler using Semantic Kernel."""
        # Create a Kernel instance
        self.kernel = Kernel()
        
        # Ensure `services` dictionary exists in `Kernel`
        self.kernel.services = {}

        # Configure execution settings
        self.execution_settings = AzureChatPromptExecutionSettings()
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        # Setup AI services correctly
        self.setup_chat_services()

    def setup_chat_services(self):
        """Set up Azure OpenAI chat services properly."""
        try:
            # "mini" model for quick usage
            mini_service = AzureChatCompletion(
                deployment_name="gpt-4o-mini",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-06-01"
            )

            # "full" model for advanced usage
            full_service = AzureChatCompletion(
                deployment_name="gpt-4o",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-06-01"
            )

            # Manually store chat services
            self.kernel.services["chat"] = {
                "mini": mini_service,
                "full": full_service
            }
            
            logger.info("Registered 'mini' and 'full' chat services.")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI services: {str(e)}")
            logger.warning("Will fallback to minimal or no LLM responses if not available.")

    def is_available(self) -> bool:
        """Check if the 'mini' LLM service is available"""
        return "chat" in self.kernel.services and "mini" in self.kernel.services["chat"]

    def is_full_model_available(self) -> bool:
        """Check if the 'full' LLM service is available"""
        return "chat" in self.kernel.services and "full" in self.kernel.services["chat"]

    async def execute_prompt(self, prompt: str, use_full_model: bool = False, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Execute a prompt using the selected LLM.

        Args:
            prompt: The input text.
            use_full_model: Whether to use the full model instead of mini.
            temperature: Generation temperature.

        Returns:
            A dictionary containing response text, model, and endpoint details.
        """
        service_id = "full" if (use_full_model and self.is_full_model_available()) else "mini"
        service = self.kernel.services["chat"].get(service_id)

        if not service:
            return {
                "text": "No chat service available for that model.",
                "model": "None",
                "deployment": "None"
            }

        try:
            logger.info(f"LLM prompt using '{service_id}' service: {prompt[:100]}...")
            self.execution_settings.temperature = temperature

            # Create a temporary chat history
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)

            # Get LLM response
            result = await service.get_chat_message_content(
                chat_history=chat_history,
                settings=self.execution_settings,
                kernel=self.kernel  # Required for function calling or plugin usage
            )

            response_text = str(result)
            logger.info(f"LLM response ({service_id}): {response_text[:100]}...")
            return {
                "text": response_text,
                "model": service_id,
                "deployment": os.getenv("AZURE_OPENAI_ENDPOINT", "unknown")
            }

        except Exception as e:
            logger.error(f"Error in direct LLM prompt: {str(e)}")
            return {
                "text": f"Error in LLM processing: {str(e)}",
                "model": "error",
                "deployment": "none"
            }
