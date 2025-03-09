"""
Pure LLM-based Intent Classification Service for Medical Assistant
"""
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class IntentClassificationService:
    """Pure LLM-based intent classification service with enhanced symptom detection"""
    
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
                    deployment_name=os.getenv("INTENT_AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
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
        
        # Define intent options
        self.intent_options = [
            "greet",
            "inform_symptoms",
            "ask_medical_info",
            "confirm",
            "deny",
            "goodbye",
            "out_of_scope"
        ]
    
    async def classify_intent_primary(self, utterance: str) -> Optional[Dict[str, float]]:
        """
        Primary intent classification using LLM with enhanced symptom detection
        
        Args:
            utterance: The user's message
            
        Returns:
            Dictionary of intent names and confidence scores, or None if LLM call fails
        """
        if not self.chat_service:
            logger.warning("No chat service available for primary intent classification")
            return None
            
        logger.info("Performing primary LLM-based intent classification")
        
        # Enhanced intent descriptions for better symptom detection
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
        
        # Enhanced prompt with clearer rules for symptom detection
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
7. Statements about duration, severity, or changes in a condition are "inform_symptoms"
8. Descriptions of medication effects or side effects are "inform_symptoms"

Respond with a JSON object in this exact format:
{{
  "intent": "the_intent_name",
  "confidence": 0.9,
  "reasoning": "Brief explanation of why this intent was chosen"
}}

Use only the intent names from the list above. The confidence should be between 0 and 1."""
        
        try:
            logger.info(f"Primary LLM Intent Classification Request: {utterance}")
            
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
                    parsed_result = json.loads(json_str)
                    
                    # Extract intent and confidence
                    intent = parsed_result.get("intent", "").lower()
                    confidence = float(parsed_result.get("confidence", 0.5))
                    reasoning = parsed_result.get("reasoning", "No reasoning provided")
                    
                    # Validate intent
                    if intent not in self.intent_options:
                        logger.warning(f"LLM returned invalid intent: {intent}")
                        return None
                    
                    # Validate confidence
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # Create score dictionary
                    scores = {i: 0.1 for i in self.intent_options}
                    scores[intent] = confidence
                    
                    logger.info(f"Primary LLM Intent Classification: {intent} ({confidence:.2f}) - {reasoning}")
                    return scores
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response: {e}")
                    return None
            else:
                logger.warning("No JSON found in LLM response")
                return None
                
        except Exception as e:
            logger.error(f"Error in primary LLM intent classification: {str(e)}")
            return None
    
    async def analyze_for_symptoms(self, utterance: str) -> Optional[float]:
        """
        Secondary LLM call specifically to detect if the message contains symptoms
        
        Args:
            utterance: The user's message
            
        Returns:
            Confidence score for symptom detection or None if LLM call fails
        """
        if not self.chat_service:
            logger.warning("No chat service available for symptom analysis")
            return None
            
        logger.info("Performing specialized symptom detection")
        
        prompt = f"""You are a medical assistant specifically analyzing if a user message contains ANY symptom information.

User message: "{utterance}"

A "symptom" includes:
- ANY physical sensation or discomfort
- ANY mental/emotional symptom or distress
- General statements about not feeling well
- Vague health issues or concerns
- Information about medication effects or side effects
- Descriptions of pain, discomfort, or abnormal sensations
- Information about duration, frequency, or patterns of health issues
- Worsening or improvement of a condition

Analyze if the message contains ANY symptom information and respond with a JSON object:
{{
  "contains_symptoms": true or false,
  "confidence": 0.9,
  "explanation": "Brief explanation of your analysis"
}}

The confidence should be between 0 and 1, with higher values for clearer symptom descriptions."""
        
        try:
            # Create a temporary chat history for symptom analysis
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
                logger.warning("Empty response from symptom analysis")
                return None
                
            logger.info("Symptom analysis response received")
            
            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_result = json.loads(json_str)
                    
                    contains_symptoms = parsed_result.get("contains_symptoms", False)
                    confidence = float(parsed_result.get("confidence", 0.5))
                    explanation = parsed_result.get("explanation", "No explanation provided")
                    
                    logger.info(f"Symptom analysis result: contains_symptoms={contains_symptoms} ({confidence:.2f}) - {explanation}")
                    
                    if contains_symptoms:
                        return confidence
                    else:
                        return 0.0
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from symptom analysis response: {e}")
                    return None
            else:
                logger.warning("No JSON found in symptom analysis response")
                return None
                
        except Exception as e:
            logger.error(f"Error in symptom analysis: {str(e)}")
            return None
    
    async def classify_intent(self, utterance: str) -> Dict[str, float]:
        """
        Multi-step intent classification using multiple LLM calls if needed
        
        Args:
            utterance: The user's message
        
        Returns:
            Dictionary of intent names and confidence scores
        """
        if not utterance.strip():
            return {"out_of_scope": 1.0}
        
        # Step 1: Try primary intent classification
        primary_result = await self.classify_intent_primary(utterance)
        
        if primary_result:
            # Check if the primary classification already detected symptoms
            top_intent = max(primary_result.items(), key=lambda x: x[1])[0]
            top_score = primary_result[top_intent]
            
            if top_intent == "inform_symptoms" and top_score > 0.5:
                logger.info(f"Primary classification detected symptoms with confidence {top_score:.2f}")
                return primary_result
                
            if top_score > 0.7:
                logger.info(f"Primary classification detected {top_intent} with high confidence {top_score:.2f}")
                return primary_result
        
        # Step 2: If primary classification failed or was uncertain, do a specialized symptom check
        symptom_confidence = await self.analyze_for_symptoms(utterance)
        
        if symptom_confidence is not None and symptom_confidence > 0.5:
            logger.info(f"Specialized symptom check detected symptoms with confidence {symptom_confidence:.2f}")
            # Create a result with high symptom confidence
            result = {intent: 0.1 for intent in self.intent_options}
            result["inform_symptoms"] = symptom_confidence
            return result
            
        # Step 3: If we still don't have a good classification, fall back to the original result or default
        if primary_result:
            logger.info("Using primary classification result as fallback")
            return primary_result
            
        # Final fallback if all LLM calls failed
        logger.warning("All LLM classification attempts failed, using default fallback")
        fallback = self._create_fallback_intent_scores(utterance)
        return fallback
    
    def _create_fallback_intent_scores(self, utterance: str) -> Dict[str, float]:
        """
        Create a minimal fallback intent classification when LLM calls fail
        
        Args:
            utterance: The user's message
            
        Returns:
            Dictionary of intent scores based on basic rules
        """
        # Set default scores
        scores = {intent: 0.1 for intent in self.intent_options}
        
        # Set out_of_scope as most likely, but with low confidence
        scores["out_of_scope"] = 0.4
        
        # Very minimal fallback checks - only for critical functionality
        if "?" in utterance:
            scores["ask_medical_info"] = 0.5
        elif any(word in utterance.lower() for word in ["yes", "yeah", "correct"]):
            scores["confirm"] = 0.8
        elif any(word in utterance.lower() for word in ["no", "nope", "not"]):
            scores["deny"] = 0.8
        elif any(word in utterance.lower() for word in ["bye", "goodbye", "thanks", "thank you"]):
            scores["goodbye"] = 0.8
        elif any(greeting in utterance.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
            scores["greet"] = 0.8
            
        return scores