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
    Handles interactions with language models using Semantic Kernel 1.23+ style.
    """

    def __init__(self):
        """Initialize the LLM handler"""
        # Create a Kernel instance.
        self.kernel = Kernel()

        # Configure the prompt execution settings.
        self.execution_settings = AzureChatPromptExecutionSettings()
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        # Register chat services using the updated API.
        self.setup_chat_services()

    def setup_chat_services(self):
        """Set up Azure OpenAI chat services and register them with the kernel."""
        try:
            # "mini" model for quick usage.
            mini_service = AzureChatCompletion(
                deployment_name="gpt-4o-mini",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-06-01"
            )
            # Register by directly assigning to the kernel's chat_services dictionary.
            self.kernel.chat_services["mini"] = mini_service
            logger.info("Registered 'mini' chat service with deployment: gpt-4o-mini")

            # "full" model for advanced usage.
            full_service = AzureChatCompletion(
                deployment_name="gpt-4o",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-06-01"
            )
            self.kernel.chat_services["full"] = full_service
            logger.info("Registered 'full' chat service with deployment: gpt-4o")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI services: {str(e)}")
            logger.warning("Will fallback to minimal or no LLM responses if not available.")

    def is_available(self) -> bool:
        """Check if the 'mini' LLM service is available"""
        mini_service = self.kernel.chat_services.get("mini")
        return mini_service is not None

    def is_full_model_available(self) -> bool:
        """Check if the 'full' LLM service is available"""
        full_service = self.kernel.chat_services.get("full")
        return full_service is not None

    async def execute_prompt(self, prompt: str, use_full_model: bool = False, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Execute a direct prompt to the chosen LLM service.

        Args:
            prompt: The text prompt to send to the LLM.
            use_full_model: Whether to use the 'full' model instead of 'mini'.
            temperature: The temperature setting for generation.

        Returns:
            Dict containing the response text, model info, and endpoint deployment.
        """
        # Decide which service ID to use.
        service_id = "full" if (use_full_model and self.is_full_model_available()) else "mini"
        # Retrieve the service from the chat_services dictionary.
        service = self.kernel.chat_services.get(service_id)
        if not service:
            return {
                "text": "No chat service available for that model.",
                "model": "None",
                "deployment": "None"
            }

        try:
            logger.info(f"LLM prompt using '{service_id}' service: {prompt[:100]}...")
            self.execution_settings.temperature = temperature

            # Create a temporary chat history for this prompt.
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)

            # Get an LLM response from the chosen chat service.
            result = await service.get_chat_message_content(
                chat_history=chat_history,
                settings=self.execution_settings,
                kernel=self.kernel  # required for function-calling or plugin usage
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

    async def calculate_diagnosis_confidence(self, symptoms: str) -> Dict[str, Any]:
        """
        Calculate confidence in diagnosis through self-reflection, using the 'full' model if available.

        Args:
            symptoms: A string describing the patient's symptoms.

        Returns:
            Dictionary with confidence score, reasoning, model info, and deployment info.
        """
        if not self.is_available() or not symptoms or symptoms == "unknown symptoms":
            return {
                "confidence": 0.0,
                "reasoning": "Insufficient information provided",
                "model": "none",
                "deployment": "none"
            }

        try:
            prompt = f"""I need to assess my confidence in diagnosing a medical condition based on these symptoms:
"{symptoms}"

Perform a self-reflection analysis and determine:
1. How specific are these symptoms? (Specific symptoms = higher confidence)
2. Are there multiple potential causes? (More potential causes = lower confidence)
3. Is there enough information for a decent diagnosis? (More data = higher confidence)
4. How severe are these symptoms? (More severe = higher threshold required)

Finally, estimate the confidence in making a diagnosis from 0.0 to 1.0.
Return ONLY a JSON object in this format:
{{
  "confidence": 0.7,
  "reasoning": "Short explanation"
}}"""

            # Use the full model if available, else fallback to 'mini'.
            response_data = await self.execute_prompt(prompt, use_full_model=True, temperature=0.3)
            response_text = response_data.get("text", "")

            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                    confidence = float(result.get("confidence", 0.0))
                    reasoning = result.get("reasoning", "No reasoning")
                    confidence = max(0.0, min(confidence, 1.0))

                    logger.info(f"Diagnosis confidence: {confidence:.2f} - {reasoning}")
                    return {
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "model": response_data.get("model", "unknown"),
                        "deployment": response_data.get("deployment", "unknown")
                    }
                except (json.JSONDecodeError, ValueError) as ex:
                    logger.error(f"Error parsing confidence result: {str(ex)}")

            # Fallback if no valid JSON object was found in the LLM's answer.
            symptom_count = len(symptoms.split(','))
            fallback_confidence = min(0.3 + (symptom_count * 0.1), 0.7)
            fallback_reasoning = "Based on symptom count only."
            logger.warning(f"Fallback confidence: {fallback_confidence:.2f}")

            return {
                "confidence": fallback_confidence,
                "reasoning": fallback_reasoning,
                "model": response_data.get("model", "unknown"),
                "deployment": response_data.get("deployment", "unknown")
            }

        except Exception as e:
            logger.error(f"Error in confidence calculation: {str(e)}")
            return {
                "confidence": 0.2,
                "reasoning": f"Error in LLM or code: {str(e)}",
                "model": "error",
                "deployment": "none"
            }
