"""
LLM handler for medical assistant bot
"""
import os
import json
import re
import logging
from typing import Optional, Dict, Any, Tuple

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles interactions with language models"""
    
    def __init__(self):
        """Initialize the LLM handler"""
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        
        # Configure execution settings
        self.execution_settings = AzureChatPromptExecutionSettings()
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        
        # Initialize chat service
        self.chat_service = None
        self.full_model_service = None
        self.setup_chat_services()
        
    def setup_chat_services(self):
        """Set up Azure OpenAI chat services"""
        try:
            # Set up the main service (mini model for efficiency)
            self.chat_service = AzureChatCompletion(
                deployment_name="gpt-4o",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2025-01-01-preview"
            )
            self.kernel.add_service(self.chat_service, service_id="mini")
            logger.info(f"Added Azure OpenAI mini service with deployment: o3-mini")
            
            # Set up the full model service for critical operations
            self.full_model_service = AzureChatCompletion(
                deployment_name="gpt-4o",
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2025-01-01-preview"
            )
            self.kernel.add_service(self.full_model_service, service_id="full")
            logger.info(f"Added Azure OpenAI full service with deployment: o1")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI services: {str(e)}")
            logger.warning("The bot will continue with fallback responses instead of actual LLM calls")
            self.chat_service = None
            self.full_model_service = None
    
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        return self.chat_service is not None
    
    def is_full_model_available(self) -> bool:
        """Check if the full model service is available"""
        return self.full_model_service is not None
    
    async def execute_prompt(self, prompt: str, use_full_model: bool = False, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Execute a direct prompt to the LLM
        
        Args:
            prompt: The prompt to send to the LLM
            use_full_model: Whether to use the full model instead of mini
            temperature: The temperature setting for generation
            
        Returns:
            Dictionary containing the response text and model info
        """
        # Select the appropriate service
        service = self.full_model_service if use_full_model and self.is_full_model_available() else self.chat_service
        
        if not service:
            return {
                "text": "LLM service not available.",
                "model": "None",
                "deployment": "None"
            }
            
        try:
            model_type = "full" if use_full_model else "mini"
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o") if use_full_model else os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME", "gpt-4o-mini")
            
            logger.info(f"Direct LLM prompt ({model_type}): {prompt[:100]}...")
            
            # Set temperature
            self.execution_settings.temperature = temperature
            
            # Create a temp chat history for this prompt
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Get LLM response
            result = await service.get_chat_message_content(
                chat_history=chat_history,
                settings=self.execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            logger.info(f"Direct LLM response ({model_type}): {response_text[:100]}...")
            
            return {
                "text": response_text,
                "model": model_type,
                "deployment": deployment_name
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
        Calculate confidence in diagnosis through self-reflection
        
        Args:
            symptoms: String containing all symptoms
            
        Returns:
            Dictionary with confidence score, reasoning, and model info
        """
        if not self.is_available() or not symptoms or symptoms == "unknown symptoms":
            return {
                "confidence": 0.0,
                "reasoning": "Insufficient information",
                "model": "none",
                "deployment": "none"
            }
        
        try:
            # Create a prompt for self-reflection
            prompt = f"""I need to assess my confidence in diagnosing a medical condition based on these symptoms:
"{symptoms}"

Perform a self-reflection analysis and determine:
1. How specific are these symptoms? (Specific symptoms increase confidence)
2. Are there multiple potential causes for these symptoms? (More potential causes decreases confidence)
3. Is there enough information to make a reasonable diagnosis? (More information increases confidence)
4. How severe or concerning are these symptoms? (Higher severity requires higher confidence)

Based on this analysis, estimate your confidence in making a diagnosis on a scale from 0.0 to 1.0.
0.0 = Complete uncertainty (impossible to diagnose)
0.5 = Moderate confidence (several potential diagnoses)
1.0 = Very high confidence (clear diagnostic pattern)

Return ONLY a JSON object with this format:
{{
  "confidence": 0.7,
  "reasoning": "Brief explanation of confidence level"
}}"""

            # Use full model for confidence calculation if available
            response_data = await self.execute_prompt(prompt, use_full_model=True, temperature=0.3)
            response_text = response_data.get("text", "")
            
            # Try to find a JSON object in the response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                    confidence = float(result.get("confidence", 0.0))
                    reasoning = result.get("reasoning", "No reasoning provided")
                    
                    # Ensure confidence is between 0 and 1
                    confidence = max(0.0, min(confidence, 1.0))
                    
                    logger.info(f"Diagnosis confidence: {confidence:.2f} - {reasoning}")
                    
                    # Return confidence with model info
                    return {
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "model": response_data.get("model", "unknown"),
                        "deployment": response_data.get("deployment", "unknown")
                    }
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing confidence result: {str(e)}")
            
            # Fallback: estimate based on symptom count
            symptom_count = len(symptoms.split(','))
            fallback_confidence = min(0.3 + (symptom_count * 0.1), 0.7)
            fallback_reasoning = "Based on the number of symptoms provided"
            
            logger.warning(f"Using fallback confidence: {fallback_confidence:.2f}")
            
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
                "reasoning": f"Error: {str(e)}",
                "model": "error",
                "deployment": "none"
            }