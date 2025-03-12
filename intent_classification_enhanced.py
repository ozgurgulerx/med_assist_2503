import logging
import json
import re
from typing import Dict, Any, List, Optional

# Import the updated function handler
from llm_function_handler_v3 import LLMFunctionHandler

logger = logging.getLogger(__name__)

class IntentClassificationService:
    """
    Enhanced intent classification service that uses function calling 
    for more reliable, consistent classification.
    """
    
    def __init__(self):
        """Initialize the intent classification service."""
        self.llm_handler = LLMFunctionHandler()
        self.register_intent_functions()
    
    def register_intent_functions(self):
        """Register semantic functions for intent classification."""
        # Create intent plugin
        intent_plugin = self.llm_handler.kernel.create_plugin("intent", "Intent classification functions")
        
        # Add function for intent classification
        self.llm_handler.kernel.add_function_to_plugin(
            plugin=intent_plugin,
            function=self.llm_handler.kernel.create_semantic_function(
                prompt="""You are an intent classifier for a medical assistant bot. 
                Classify the user's message into one of these intents:\n
                1. symptomReporting - User is reporting symptoms or health concerns\n
                2. medicalInquiry - User is asking for medical information or advice\n
                3. smallTalk - User is making small talk, casual conversation\n
                4. greeting - User is greeting the bot\n
                5. farewell - User is saying goodbye\n
                6. confirm - User is confirming or agreeing with something\n
                7. deny - User is denying or disagreeing with something\n
                8. endConversation - User wants to end the conversation\n
                9. emergency - User is reporting a medical emergency\n
                10. out_of_scope - User's message is not related to health or cannot be processed\n\n
                User message: {{$message}}\n
                User context: {{$context}}\n\n
                Return a JSON object with:\n
                \"intent\": the most likely intent from the list above,\n
                \"confidence\": a confidence score between 0.0-1.0\n
                \"reasoning\": a brief explanation of why this intent was chosen""",
                function_name="classify_intent",
                description="Classifies user messages into predefined intents"
            )
        )
        
        # Add function for extracting intents from conversational messages
        self.llm_handler.kernel.add_function_to_plugin(
            plugin=intent_plugin,
            function=self.llm_handler.kernel.create_semantic_function(
                prompt="""Extract the medical intents from this user message.\n\n
                User message: {{$message}}\n
                Previous conversation context: {{$context}}\n\n
                Identify all possible intents in the message, focusing on:\n
                1. Is the user reporting symptoms? If so, what symptoms?\n
                2. Is the user asking for medical information? What specific information?\n
                3. Is the user confirming or denying something in response to a previous question?\n
                4. Is the user showing signs of a medical emergency?\n\n
                Return a JSON object with:\n
                \"primary_intent\": the main purpose of the message,\n
                \"detected_intents\": an array of all detected intents,\n
                \"extracted_entities\": any medical entities mentioned (symptoms, conditions, etc.)""",
                function_name="extract_intents",
                description="Extracts multiple intents and entities from user messages"
            )
        )
        
        logger.info("Successfully registered intent classification functions")
    
    async def classify_intent(self, message: str, context: str = "") -> Dict[str, Any]:
        """
        Classify the intent of a user message using function calling.
        
        Args:
            message: The user's message
            context: Optional context about the conversation
            
        Returns:
            Dictionary with intent, confidence, and reasoning
        """
        try:
            logger.info(f"Classifying intent for message: '{message[:50]}...' with context: '{context[:50]}...'")
            
            # Use the intent classification function
            result = await self.llm_handler.invoke_function(
                "intent.classify_intent",
                {
                    "message": message,
                    "context": context
                },
                service_id="mini"  # Use the fastest model for intent classification
            )
            
            response_text = result.get("text", "")
            
            # Extract JSON from response if needed
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group(0))
            else:
                # If no JSON structure, create a default one
                intent_data = {"intent": "medicalInquiry", "confidence": 0.6, "reasoning": "Fallback classification"}
            
            logger.info(f"Classified intent: {intent_data.get('intent', 'unknown')} with confidence: {intent_data.get('confidence', 0.0)}")
            return intent_data
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            # Return a default intent if classification fails
            return {"intent": "medicalInquiry", "confidence": 0.6, "reasoning": f"Error in classification: {str(e)}"}
    
    async def extract_multiple_intents(self, message: str, context: str = "") -> Dict[str, Any]:
        """
        Extract multiple intents and medical entities from a user message.
        Useful for complex messages that might contain multiple intentions.
        
        Args:
            message: The user's message
            context: Optional context about the conversation
            
        Returns:
            Dictionary with primary intent, all detected intents, and extracted entities
        """
        try:
            # Use the intent extraction function
            result = await self.llm_handler.invoke_function(
                "intent.extract_intents",
                {
                    "message": message,
                    "context": context
                },
                service_id="mini"
            )
            
            response_text = result.get("text", "")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                intents_data = json.loads(json_match.group(0))
            else:
                # Default response if no JSON structure found
                intents_data = {
                    "primary_intent": "medicalInquiry",
                    "detected_intents": ["medicalInquiry"],
                    "extracted_entities": []
                }
            
            return intents_data
            
        except Exception as e:
            logger.error(f"Error extracting multiple intents: {str(e)}")
            # Return a default response if extraction fails
            return {
                "primary_intent": "medicalInquiry",
                "detected_intents": ["medicalInquiry"],
                "extracted_entities": []
            }
