"""
Pure LLM-based Intent Classification Service for Medical Assistant
"""
import os
import re
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class IntentClassificationService:
    """Pure LLM-based intent classification service with minimal fallbacks"""
    
    def __init__(self, chat_service: AzureChatCompletion = None, kernel: Kernel = None):
        """
        Initialize the intent classifier
        
        Args:
            chat_service: Azure Chat Completion service for LLM-based classification (optional)
            kernel: Semantic Kernel instance for plugin access (optional)
        """
        # Create a dedicated kernel and chat service for intent classification if not provided
        if kernel is None:
            self.kernel = Kernel()
            self.is_dedicated_kernel = True
        else:
            self.kernel = kernel
            self.is_dedicated_kernel = False
            
        # Create a dedicated Azure OpenAI client for intent classification if not provided
        if chat_service is None:
            try:
                self.chat_service = AzureChatCompletion(
                    deployment_name=os.getenv("INTENT_AZURE_OPENAI_DEPLOYMENT_NAME"),
                    endpoint=os.getenv("INTENT_AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")),
                    api_key=os.getenv("INTENT_AZURE_OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")),
                    api_version="2024-06-01"
                )
                self.kernel.add_service(self.chat_service)
                logger.info("Created Azure OpenAI service for intent classification")
                self.is_dedicated_service = True
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI service for intent classification: {str(e)}")
                logger.warning("Intent classifier will use minimal fallbacks only")
                self.chat_service = None
                self.is_dedicated_service = False
        else:
            self.chat_service = chat_service
            self.is_dedicated_service = False
        
        # Define intent options - we'll need these regardless
        self.intent_options = [
            "greet",
            "inform_symptoms",
            "ask_medical_info",
            "confirm",
            "deny",
            "goodbye",
            "out_of_scope"
        ]
    
    async def classify_intent_with_llm(self, utterance: str) -> Optional[Dict[str, float]]:
        """
        Classify intent using LLM
        
        Args:
            utterance: The user's message
            
        Returns:
            Dictionary of intent names and confidence scores, or None if LLM call fails
        """
        if not self.chat_service:
            logger.warning("No chat service available, cannot use LLM for intent classification")
            return None
            
        logger.info("Attempting LLM-based intent classification")
        
        # Intent descriptions
        intent_descriptions = {
            "greet": "Greetings and introductions",
            "inform_symptoms": "Descriptions of ANY physical or mental symptoms, health issues, medical conditions, pain, discomfort, or abnormal sensations. This includes general descriptions of not feeling well or having issues with bodily functions.",
            "ask_medical_info": "Questions about medical topics, treatments, medications, or health information",
            "confirm": "Affirmative responses (yes, correct, etc.)",
            "deny": "Negative responses (no, incorrect, etc.)",
            "goodbye": "Ending the conversation or expressing thanks",
            "out_of_scope": "Messages that don't fit into any other category"
        }
        
        # Format the intent options for the prompt
        intent_options_text = "\n".join([f"- {intent}: {desc}" for intent, desc in intent_descriptions.items()])
        
        prompt = f"""You are a medical assistant classifying the intent of a user message.
Given the user message, determine which of the following intents best matches:

{intent_options_text}

User message: "{utterance}"

IMPORTANT MEDICAL CLASSIFICATION RULES:
1. ANY mention of physical symptoms, pain, illness, or health problems should be classified as "inform_symptoms"
2. Even VAGUE descriptions like "not feeling well" or "having issues" should be classified as "inform_symptoms"
3. Follow-up details about previously mentioned symptoms are still "inform_symptoms"
4. Medical questions are "ask_medical_info", but descriptions of one's own condition are "inform_symptoms"
5. When in doubt between symptom descriptions and medical questions, prioritize "inform_symptoms"
6. Responses to direct questions about symptoms should be classified as "inform_symptoms"

Respond with a JSON object in this exact format:
{{
  "intent": "the_intent_name",
  "confidence": 0.9,
  "reasoning": "Brief explanation of why this intent was chosen"
}}

Use only the intent names from the list above. The confidence should be between 0 and 1."""
        
        try:
            logger.info(f"LLM Intent Classification Request: {utterance}")
            
            # Create a temporary chat history for intent classification
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Configure execution settings
            execution_settings = AzureChatPromptExecutionSettings()
            
            # Get LLM response
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            
            if not response_text:
                logger.warning("Empty response from LLM")
                return None
                
            logger.info(f"LLM Intent Classification Response received")
            
            # Parse the JSON response
            # Extract JSON from response, handling potential text around it
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    
                    # Extract intent and confidence
                    intent = result.get("intent", "").lower()
                    confidence = float(result.get("confidence", 0.5))
                    
                    # Validate intent
                    if intent not in self.intent_options:
                        logger.warning(f"LLM returned invalid intent: {intent}")
                        return self._create_fallback_intent_scores()
                    
                    # Validate confidence
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # Create score dictionary
                    scores = {i: 0.1 for i in self.intent_options}
                    scores[intent] = confidence
                    
                    logger.info(f"LLM Intent Classification Result: {intent} with confidence {confidence}")
                    return scores
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response: {e}")
                    return None
            else:
                logger.warning("No JSON found in LLM response")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {str(e)}")
            return None
    
    def _create_fallback_intent_scores(self) -> Dict[str, float]:
        """Create a minimal fallback intent classification"""
        # Set out_of_scope as most likely, but with low confidence
        scores = {intent: 0.1 for intent in self.intent_options}
        scores["out_of_scope"] = 0.4
        return scores
    
    async def classify_intent(self, utterance: str) -> Dict[str, float]:
        """
        Classify the intent of a user utterance
        
        Args:
            utterance: The user's message
        
        Returns:
            Dictionary of intent names and confidence scores
        """
        if not utterance.strip():
            return {"out_of_scope": 1.0}
        
        # Use LLM classification - this is the primary method
        llm_result = await self.classify_intent_with_llm(utterance)
        
        if llm_result:
            return llm_result
        
        # If LLM classification fails, use minimal fallback logic
        # Just a very basic check for common patterns
        result = self._create_fallback_intent_scores()
        
        # Very minimal fallback checks - only for critical functionality
        if any(greeting in utterance.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
            result["greet"] = 0.8
        elif "?" in utterance:
            result["ask_medical_info"] = 0.6
        elif any(word in utterance.lower() for word in ["yes", "yeah", "correct"]):
            result["confirm"] = 0.8
        elif any(word in utterance.lower() for word in ["no", "nope", "not"]):
            result["deny"] = 0.8
        elif any(word in utterance.lower() for word in ["bye", "goodbye", "thanks", "thank you"]):
            result["goodbye"] = 0.8
            
        return result