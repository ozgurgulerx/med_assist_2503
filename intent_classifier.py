"""
Intent Classification Service for Medical Assistant
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
    """Service for classifying user intents using LLM when available, with pattern matching as fallback"""
    
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
                    endpoint=os.getenv("INTENT_AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("INTENT_AZURE_OPENAI_API_KEY"),
                    api_version="2024-06-01"
                )
                self.kernel.add_service(self.chat_service)
                logger.info("Created dedicated Azure OpenAI service for intent classification")
                self.is_dedicated_service = True
            except Exception as e:
                logger.error(f"Failed to initialize dedicated Azure OpenAI service for intent classification: {str(e)}")
                logger.warning("Intent classifier will use pattern matching as fallback")
                self.chat_service = None
                self.is_dedicated_service = False
        else:
            self.chat_service = chat_service
            self.is_dedicated_service = False
        
        self.intent_options = [
            "greet",
            "inform_symptoms",
            "ask_medical_info",
            "confirm",
            "deny",
            "goodbye",
            "out_of_scope"
        ]
        
        # Enhanced pattern definitions for fallback
        self.patterns = {
            "greet": [
                r"\b(?:hello|hi|hey|greetings|good\s*(?:morning|afternoon|evening))\b"
            ],
            "inform_symptoms": [
                r"\b(?:headache|pain|ache|hurt|sore|dizzy|nausea|vomit|fever|cough|sneeze|rash|swelling|breathing|tired|fatigue|exhausted|symptom|sick|ill|unwell)\b",
                r"\b(?:feeling|experienc(?:e|ing)|suffer(?:ing)?|having)\b",
                r"\b(?:since|started|begin|began|worse|better|worsen|improve)\b",
                r"\b(?:morning|night|day|week|daily|constant|persistent|chronic|acute)\b"
            ],
            "ask_medical_info": [
                r"(?:what|how|why|when|where|which|can|should)\s.+(?:\?|$)",
                r"\b(?:tell me about|explain|information|advice|recommend|suggest)\b",
                r"\b(?:treatment|medicine|medication|drug|therapy|doctor|normal|average|typical|blood pressure|cholesterol|diabetes|heart)\b"
            ],
            "confirm": [
                r"\b(?:yes|yeah|yep|correct|right|exactly|agree|true|affirmative)\b"
            ],
            "deny": [
                r"\b(?:no|nope|not|don't|doesn't|didn't|haven't|hasn't|can't|none|never)\b"
            ],
            "goodbye": [
                r"\b(?:bye|goodbye|farewell|thanks|thank you|appreciate|good night)\b"
            ]
        }
        
        # Compile all patterns
        self.compiled_patterns = {}
        for intent, pattern_list in self.patterns.items():
            self.compiled_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        
    async def classify_intent_with_llm(self, utterance: str) -> Optional[Dict[str, float]]:
        """
        Classify intent using LLM with the Semantic Kernel
        
        Args:
            utterance: The user's message
            
        Returns:
            Dictionary of intent names and confidence scores, or None if LLM call fails
        """
        if not self.chat_service:
            logger.warning("No chat service available, cannot use LLM for intent classification")
            return None
            
        logger.info("Attempting LLM-based intent classification")
        
        # Create a description of each intent for the LLM
        intent_descriptions = {
            "greet": "Greetings and introductions",
            "inform_symptoms": "Descriptions of symptoms, health issues, or medical conditions",
            "ask_medical_info": "Questions about medical topics or health information",
            "confirm": "Affirmative responses (yes, correct, etc.)",
            "deny": "Negative responses (no, incorrect, etc.)",
            "goodbye": "Ending the conversation or expressing thanks",
            "out_of_scope": "Messages that don't fit into any other category"
        }
        
        # Format the intent options for the prompt
        intent_options_text = "\n".join([f"- {intent}: {desc}" for intent, desc in intent_descriptions.items()])
        
        prompt = f"""You are classifying the intent of a user message for a medical chatbot.
Given the user message, determine which of the following intents best matches:

{intent_options_text}

User message: "{utterance}"

Important: When a user mentions ANY health issues, symptoms, pain, or physical/mental conditions, classify it as "inform_symptoms".
If they ask ANY questions about medical information, treatments, or conditions, classify it as "ask_medical_info".

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
            
            # Get LLM response using the working method
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            
            if not response_text:
                logger.warning("Empty response from LLM, falling back to pattern matching")
                return None
                
            logger.info(f"LLM Intent Classification Response: {response_text}")
            
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
                        logger.warning(f"LLM returned invalid intent: {intent}, falling back")
                        return None
                    
                    # Validate confidence
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # Create score dictionary
                    scores = {i: 0.1 for i in self.intent_options}
                    scores[intent] = confidence
                    
                    logger.info(f"LLM Intent Classification Result: {intent} with confidence {confidence}")
                    return scores
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response: {e}")
                    logger.error(f"Response was: {json_str}")
                    return None
            else:
                logger.warning("No JSON found in LLM response")
                return None
                
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {str(e)}")
            return None
    
    async def classify_intent_with_patterns(self, utterance: str) -> Dict[str, float]:
        """
        Classify intent using pattern matching as fallback
        
        Args:
            utterance: The user's message
            
        Returns:
            Dictionary of intent names and confidence scores
        """
        logger.info("Using pattern-based intent classification")
        
        # Initialize scores for all intents
        scores = {intent: 0.0 for intent in self.intent_options}
        scores["out_of_scope"] = 0.1  # Default low score for out_of_scope
        
        # Calculate scores based on pattern matches
        for intent, patterns in self.compiled_patterns.items():
            matches = sum(1 for pattern in patterns if pattern.search(utterance))
            if matches > 0:
                # Calculate score based on number of matches
                # More matches = higher confidence
                base_score = 0.5
                match_score = min(matches * 0.15, 0.4)  # Up to 0.4 for multiple matches
                scores[intent] = base_score + match_score
        
        # Special case: if inform_symptoms has evidence but not very strong,
        # and the text is long enough, increase its score
        if 0.2 < scores["inform_symptoms"] < 0.7 and len(utterance.split()) > 5:
            scores["inform_symptoms"] += 0.15
        
        # If asking a question, boost ask_medical_info
        if "?" in utterance:
            scores["ask_medical_info"] = max(scores["ask_medical_info"], 0.75)
        
        # If no intent has significant confidence, boost out_of_scope
        if all(score < 0.4 for intent, score in scores.items() if intent != "out_of_scope"):
            scores["out_of_scope"] = 0.5
        
        # Log the scores
        logger.info(f"Pattern-based intent scores: {scores}")
        
        return scores
    
    async def classify_intent(self, utterance: str) -> Dict[str, float]:
        """
        Classify the intent of a user utterance using LLM first, then pattern matching as fallback
        
        Args:
            utterance: The user's message
        
        Returns:
            Dictionary of intent names and confidence scores
        """
        if not utterance.strip():
            return {"out_of_scope": 1.0}
        
        # Try LLM classification first
        llm_result = await self.classify_intent_with_llm(utterance)
        
        if llm_result:
            return llm_result
            
        # Fall back to pattern-based classification if LLM fails
        return await self.classify_intent_with_patterns(utterance)