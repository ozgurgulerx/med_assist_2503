"""
Pure LLM-Based Medical Intent Classification System
"""
import os
import re
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
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


class DialogueStateTracker:
    """Tracks the state of the dialogue to provide context for intent classification."""
    
    def __init__(self):
        """Initialize the dialogue state tracker."""
        # Track conversation state for each user
        self.user_states = {}
        
        # Track expected response types
        self.expected_responses = {}
        
        # Define possible states
        self.states = [
            "greeting",
            "collecting_symptoms",
            "asking_followup",
            "providing_info",
            "verification",
            "closing"
        ]
    
    def get_state(self, user_id: str) -> str:
        """Get the current dialogue state for a user."""
        return self.user_states.get(user_id, "greeting")
    
    def update_state(self, user_id: str, state: str) -> None:
        """Update the dialogue state for a user."""
        if state in self.states:
            self.user_states[user_id] = state
    
    def set_expected_response(self, user_id: str, expected_type: str) -> None:
        """
        Set the type of response expected from the user.
        
        Args:
            user_id: The user's identifier.
            expected_type: Type of expected response (e.g., "yes_no", "symptom_details").
        """
        self.expected_responses[user_id] = expected_type
    
    def get_expected_response(self, user_id: str) -> Optional[str]:
        """Get the type of response expected from the user."""
        return self.expected_responses.get(user_id)
    
    async def analyze_assistant_message(self, user_id: str, message: str, llm_service) -> None:
        """
        Update dialogue state based on assistant's message using LLM analysis.
        
        Args:
            user_id: The user's identifier.
            message: The assistant's message.
            llm_service: LLM service for analysis.
        """
        if not llm_service:
            # Default to collecting symptoms if no LLM available
            self.update_state(user_id, "collecting_symptoms")
            return
            
        try:
            prompt = f"""Analyze this assistant message in a medical conversation to determine the dialogue state.

Assistant message: "{message}"

Based only on this message, identify what state the conversation is in:
1. Is the assistant greeting the user? (greeting)
2. Is the assistant asking about or exploring symptoms? (collecting_symptoms)
3. Is the assistant asking follow-up questions about symptoms? (asking_followup)
4. Is the assistant providing medical information? (providing_info)
5. Is the assistant verifying or confirming symptoms? (verification)
6. Is the assistant ending the conversation? (closing)

Also determine what type of response is expected from the user:
- Is a yes/no response expected? (yes_no)
- Is detailed symptom information expected? (symptom_details)
- Is an acknowledgment expected? (acknowledgment)
- Is a farewell expected? (farewell)
- Is an open-ended response expected? (open_ended)

Respond with a JSON object:
{{
  "dialogue_state": "greeting|collecting_symptoms|asking_followup|providing_info|verification|closing",
  "expected_response": "yes_no|symptom_details|acknowledgment|farewell|open_ended",
  "reasoning": "Brief explanation"
}}"""

            # Create temporary chat history
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Configure execution settings
            execution_settings = AzureChatPromptExecutionSettings()
            execution_settings.temperature = 0.2
            
            # Get LLM response
            result = await llm_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings,
                kernel=None
            )
            
            response_text = str(result)
            
            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = json.loads(json_str)
                
                # Extract dialogue state and expected response
                dialogue_state = parsed_result.get("dialogue_state")
                expected_response = parsed_result.get("expected_response")
                
                if dialogue_state and dialogue_state in self.states:
                    self.update_state(user_id, dialogue_state)
                
                if expected_response:
                    self.set_expected_response(user_id, expected_response)
                
                logger.info(f"Dialogue state analysis: {dialogue_state}, expected response: {expected_response}")
                
        except Exception as e:
            logger.error(f"Error analyzing assistant message: {str(e)}")
            # Default to collecting symptoms in case of error
            self.update_state(user_id, "collecting_symptoms")


class MedicalIntentClassifier:
    """Pure LLM-based intent classification system with no hardcoded patterns, using updated intent classes."""
    
    def __init__(self, chat_service: AzureChatCompletion = None, kernel: Kernel = None):
        """
        Initialize the intent classifier.
        
        Args:
            chat_service: Azure Chat Completion service for LLM-based classification (optional).
            kernel: Semantic Kernel instance for plugin access (optional).
        """
        # Load environment variables
        load_dotenv()
        
        # Store initialization parameters for lazy loading
        self._chat_service = chat_service
        self._kernel = kernel
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Define updated intent options (no spaces/special chars):
        self.intent_options = [
            "greeting",
            "symptomReporting",
            "symptomClarification",
            "medicalInquiry",
            "checkDiagnosis",
            "demographicInfo",
            "smallTalk",
            "out_of_scope",
            "urgentEmergency",
            "endConversation",
            "verification"
        ]
        
        # Define examples of out-of-scope messages for better classification
        self.out_of_scope_examples = [
            "What's the weather like today?",
            "Can you recommend a good restaurant?",
            "Tell me about the latest news",
            "What's your favorite movie?",
            "How do I fix my computer?",
            "Can you help me with my homework?",
            "What's the capital of France?",
            "Tell me a joke",
            "Who won the game last night?",
            "What's the best vacation spot?"
        ]
        
        # Initialize dialogue state tracker
        self.dialogue_tracker = DialogueStateTracker()
        
        # Store conversation history for context
        self.conversation_histories = {}
        self.max_history_length = 5
        
        # Initialize a cache for recent classifications to avoid repeated processing
        self.classification_cache = {}
        self.cache_ttl = 300  # seconds
        
        # Initialize previous questions cache to detect follow-up responses
        self.previous_questions = {}
        
        # Pre-populate cache with common greetings for instant response
        common_greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        for greeting in common_greetings:
            self.classification_cache[f"default_user:{greeting}"] = {"greeting": 0.95, "smallTalk": 0.05}
            
        logger.info("Intent classifier initialized with lazy loading")

    async def _ensure_initialized(self):
        """Ensure that the LLM services are initialized before use (lazy initialization)"""
        if self._is_initialized:
            return
            
        async with self._initialization_lock:
            # Check again in case another task initialized while we were waiting
            if self._is_initialized:
                return
                
            logger.info("Lazily initializing LLM services for intent classification")
            
            # Create a dedicated kernel for intent classification if not provided
            if self._kernel is None:
                self.kernel = Kernel()
                self.is_dedicated_kernel = True
            else:
                self.kernel = self._kernel
                self.is_dedicated_kernel = False
                
            # Create a dedicated Azure OpenAI client for intent classification if not provided
            if self._chat_service is None:
                try:
                    self.chat_service = AzureChatCompletion(
                        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version="2024-06-01"
                    )
                    self.kernel.add_service(self.chat_service)
                    logger.info("Created Azure OpenAI service for intent classification")
                    self.is_dedicated_service = True
                except Exception as e:
                    logger.error(f"Failed to initialize Azure OpenAI service for intent classification: {str(e)}")
                    logger.warning("Intent classifier will use fallbacks only")
                    self.chat_service = None
                    self.is_dedicated_service = False
            else:
                self.chat_service = self._chat_service
                self.is_dedicated_service = False
                
            self._is_initialized = True
            logger.info("LLM services for intent classification initialized successfully")
    
    def add_to_conversation_history(self, user_id: str, message: str, role: str = "user") -> None:
        """
        Add a message to the conversation history for a user.
        
        Args:
            user_id: The user's identifier.
            message: The message text.
            role: The role of the message sender ("user" or "assistant").
        """
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        
        # Add the new message
        self.conversation_histories[user_id].append({
            "role": role,
            "content": message
        })
        
        # Trim history if needed
        if len(self.conversation_histories[user_id]) > self.max_history_length * 2:
            self.conversation_histories[user_id] = self.conversation_histories[user_id][-self.max_history_length * 2:]
        
        # Update dialogue state if this is an assistant message
        if role == "assistant":
            asyncio.create_task(self.dialogue_tracker.analyze_assistant_message(
                user_id, message, self.chat_service
            ))
            
            # If it's a question, store it to help with follow-up classification
            if "?" in message:
                self.previous_questions[user_id] = message
    
    def get_formatted_history(self, user_id: str, max_turns: int = 3) -> str:
        """
        Get formatted conversation history for a user.
        
        Args:
            user_id: The user's identifier.
            max_turns: Maximum number of conversation turns to include.
            
        Returns:
            Formatted conversation history.
        """
        if user_id not in self.conversation_histories or not self.conversation_histories[user_id]:
            return "No previous conversation"
        
        # Get the most recent turns
        history = self.conversation_histories[user_id][-max_turns*2:]
        
        # Format the conversation history
        formatted_history = ""
        for message in history:
            role_name = "User" if message["role"] == "user" else "Assistant"
            formatted_history += f"{role_name}: {message['content']}\n"
        
        return formatted_history.strip()
    
    def get_previous_question(self, user_id: str) -> str:
        """Get the most recent question from the assistant."""
        return self.previous_questions.get(user_id, "")
    
    async def analyze_implied_symptoms(self, utterance: str, dialogue_state: str, previous_question: str) -> Optional[float]:
        """
        Analyze if a message contains implied symptoms based on dialogue context.
        
        Args:
            utterance: The user's message.
            dialogue_state: Current dialogue state.
            previous_question: Previous question from the assistant.
            
        Returns:
            Confidence score for implied symptom detection.
        """
        # Ensure LLM services are initialized
        await self._ensure_initialized()
        
        if not hasattr(self, 'chat_service') or self.chat_service is None:
            return None
        
        # Skip this analysis for very short responses to avoid unnecessary API calls
        if len(utterance.split()) <= 1:
            return None
        
        prompt = f"""You're analyzing a patient's response to determine if it contains health-related information, even if implicit.

Previous assistant question: "{previous_question}"
Current dialogue state: {dialogue_state}
Patient's response: "{utterance}"

In a medical conversation, patients often provide symptom information indirectly or implicitly.
Analyze whether this response contains ANY health-related information, even if subtle or implicit.

Respond with a JSON object:
{{
  "contains_health_info": true or false,
  "confidence": 0.9,
  "explanation": "Brief explanation"
}}"""

        try:
            # Create a temporary chat history
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Configure execution settings
            execution_settings = AzureChatPromptExecutionSettings()
            execution_settings.temperature = 0.2
            
            # Get LLM response
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            
            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_result = json.loads(json_str)
                    
                    contains_health_info = parsed_result.get("contains_health_info", False)
                    confidence = float(parsed_result.get("confidence", 0.5))
                    
                    if contains_health_info:
                        return confidence
                    else:
                        return 0.0
                    
                except json.JSONDecodeError:
                    return None
            
            return None
                
        except Exception as e:
            logger.error(f"Error analyzing implied symptoms: {str(e)}")
            return None
    
    async def classify_with_dialogue_context(self, utterance: str, user_id: str) -> Optional[Dict[str, float]]:
        """
        Classify intent with full dialogue context awareness.
        
        Args:
            utterance: The user's message.
            user_id: The user's identifier.
            
        Returns:
            Dictionary of intent names and confidence scores.
        """
        # Ensure LLM services are initialized
        await self._ensure_initialized()
        
        if not hasattr(self, 'chat_service') or self.chat_service is None:
            return {}
        
        # Get dialogue context
        dialogue_state = self.dialogue_tracker.get_state(user_id)
        expected_response = self.dialogue_tracker.get_expected_response(user_id)
        conversation_history = self.get_formatted_history(user_id)
        previous_question = self.get_previous_question(user_id)
        
        # Create a contextual prompt with updated categories
        prompt = f"""You are a medical assistant analyzing patient intent with full dialogue context.

CONVERSATION HISTORY:
{conversation_history}

DIALOGUE CONTEXT:
- Current dialogue state: {dialogue_state}
- Expected response type: {expected_response or "None"}
- Previous assistant question: "{previous_question}"

CURRENT USER MESSAGE: "{utterance}"

Given this full context, classify the user's intent into one of these categories:

- greeting
- symptomReporting
- symptomClarification
- medicalInquiry
- checkDiagnosis
- demographicInfo
- smallTalk
- out_of_scope
- urgentEmergency
- endConversation
- verification

Respond with a JSON object:
{{
  "intent": "the_intent_name",
  "confidence": 0.9,
  "reasoning": "Brief explanation with reference to dialogue context"
}}"""

        try:
            # Create a temporary chat history
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Configure execution settings
            execution_settings = AzureChatPromptExecutionSettings()
            execution_settings.temperature = 0.2
            
            # Get LLM response
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            
            # Parse the JSON response
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
                    
                    logger.info(f"Dialogue context classification: {intent} ({confidence:.2f}) - {reasoning}")
                    return scores
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from dialogue context classification: {e}")
                    return None
            else:
                logger.warning("No JSON found in dialogue context classification response")
                return None
                
        except Exception as e:
            logger.error(f"Error in dialogue context classification: {str(e)}")
            return None
    
    async def classify_with_medical_knowledge(self, utterance: str) -> Optional[Dict[str, float]]:
        """
        Classify intent using medical knowledge and terminology.
        
        Args:
            utterance: The user's message.
            
        Returns:
            Dictionary of intent names and confidence scores.
        """
        # Ensure LLM services are initialized
        await self._ensure_initialized()
        
        if not hasattr(self, 'chat_service') or self.chat_service is None:
            return {}
        
        # Updated categories
        prompt = f"""As a medical professional, analyze this patient message to determine its intent:

Patient message: "{utterance}"

Classify this message into one of these intent categories:

- greeting
- symptomReporting
- symptomClarification
- medicalInquiry
- checkDiagnosis
- demographicInfo
- smallTalk
- out_of_scope
- urgentEmergency
- endConversation
- verification

Respond with a JSON object:
{{
  "intent": "the_intent_name",
  "confidence": 0.9,
  "reasoning": "Brief explanation from a medical perspective"
}}"""

        try:
            # Create a temporary chat history
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Configure execution settings
            execution_settings = AzureChatPromptExecutionSettings()
            execution_settings.temperature = 0.2
            
            # Get LLM response
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            
            # Parse the JSON response
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
                    
                    logger.info(f"Medical knowledge classification: {intent} ({confidence:.2f}) - {reasoning}")
                    return scores
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from medical knowledge classification: {e}")
                    return None
            else:
                logger.warning("No JSON found in medical knowledge classification response")
                return None
                
        except Exception as e:
            logger.error(f"Error in medical knowledge classification: {str(e)}")
            return None
    
    async def generate_llm_fallback(self, utterance: str, user_id: str) -> Dict[str, float]:
        """
        Generate a fallback classification using direct LLM prompt.
        
        Args:
            utterance: The user's message.
            user_id: The user's identifier.
            
        Returns:
            Dictionary of intent names and confidence scores.
        """
        # Ensure LLM services are initialized
        await self._ensure_initialized()
        
        if not hasattr(self, 'chat_service') or self.chat_service is None:
            # If no LLM service is available, return a basic classification
            if len(utterance.split()) <= 3:
                return {"greeting": 0.7, "smallTalk": 0.3}
            else:
                return {"out_of_scope": 0.8, "smallTalk": 0.2}
        
        # Get minimal dialogue context
        dialogue_state = self.dialogue_tracker.get_state(user_id)
        
        # Updated categories
        prompt = f"""You're a medical assistant analyzing a single user message.

User message: "{utterance}"
Current dialogue state: {dialogue_state}

Classify which intent is most likely:

- greeting
- symptomReporting
- symptomClarification
- medicalInquiry
- checkDiagnosis
- demographicInfo
- smallTalk
- out_of_scope
- urgentEmergency
- endConversation
- verification

This is for a medical assistant, so be aware that many messages will contain health information.
Respond with a JSON object:
{{
  "intent": "the_intent_name",
  "confidence": 0.8
}}"""

        try:
            # Create a temporary chat history
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Configure execution settings
            execution_settings = AzureChatPromptExecutionSettings()
            execution_settings.temperature = 0.1
            
            # Get LLM response
            result = await self.chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=execution_settings,
                kernel=self.kernel
            )
            
            response_text = str(result)
            
            # Parse the JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = json.loads(json_str)
                
                # Extract intent and confidence
                intent = parsed_result.get("intent", "").lower()
                confidence = float(parsed_result.get("confidence", 0.5))
                
                if intent in self.intent_options:
                    # Create score dictionary
                    scores = {i: 0.1 for i in self.intent_options}
                    scores[intent] = confidence
                    return scores
            
            # If we reached here, something went wrong with parsing
            # Emergency fallback
            scores = {intent: 0.1 for intent in self.intent_options}
            scores["symptomReporting"] = 0.6
            return scores
            
        except Exception as e:
            logger.error(f"Error in LLM fallback: {str(e)}")
            # Emergency fallback
            scores = {intent: 0.1 for intent in self.intent_options}
            scores["symptomReporting"] = 0.6
            return scores
    
    async def weighted_ensemble_classification(self, methods_results: List[Tuple[str, Dict[str, float]]], 
                                              utterance: str, user_id: str) -> Dict[str, float]:
        """
        Combine results from multiple classification methods using a dynamic weighted ensemble.
        
        Args:
            methods_results: List of (method_name, result) tuples from different methods.
            utterance: The user's message.
            user_id: The user's identifier.
            
        Returns:
            Combined intent scores.
        """
        # Initialize combined scores
        combined_scores = {intent: 0.0 for intent in self.intent_options}
        
        # Get dialogue context to determine weights
        dialogue_state = self.dialogue_tracker.get_state(user_id)
        
        # Define method weights based on dialogue state
        if dialogue_state in ["collecting_symptoms", "asking_followup", "verification"]:
            # When collecting symptoms, dialogue context is most important
            weights = {
                "dialogue_context": 0.5,
                "medical_knowledge": 0.3,
                "implied_symptoms": 0.2
            }
        elif dialogue_state in ["greeting", "closing"]:
            # For simpler states, methods are roughly equal
            weights = {
                "dialogue_context": 0.4,
                "medical_knowledge": 0.3,
                "implied_symptoms": 0.3
            }
        else:
            # Default weights
            weights = {
                "dialogue_context": 0.4,
                "medical_knowledge": 0.4,
                "implied_symptoms": 0.2
            }
        
        # Track total weight applied
        total_weight = 0.0
        
        # Add weighted scores from each method
        for method_name, result in methods_results:
            if result and method_name in weights:
                method_weight = weights[method_name]
                
                if method_name == "implied_symptoms":
                    # This is just a confidence measure for "symptomReporting"
                    if isinstance(result, dict):
                        # If it's a dict, extract the symptomReporting confidence
                        confidence = result.get("symptomReporting", 0.0)
                    else:
                        # If it's a float, use it directly
                        confidence = result
                        
                    if confidence > 0.5:
                        combined_scores["symptomReporting"] += confidence * method_weight
                        total_weight += method_weight
                else:
                    # Normal classification result
                    for intent, score in result.items():
                        combined_scores[intent] += score * method_weight
                    
                    total_weight += method_weight
        
        # If no methods returned results, revert to fallback
        if total_weight < 0.1:
            logger.warning("No classification methods succeeded, using LLM fallback")
            return await self.generate_llm_fallback(utterance, user_id)
        
        # Normalize scores based on applied weight
        for intent in combined_scores:
            combined_scores[intent] = (
                combined_scores[intent] / total_weight if total_weight > 0 else 0.1
            )
        
        # Get the top intent for logging
        top_intent = max(combined_scores.items(), key=lambda x: x[1])
        logger.info(f"Ensemble classification result: {top_intent[0]} ({top_intent[1]:.2f})")
        return combined_scores
    
    async def classify_intent(self, utterance: str, user_id: str = "default_user") -> Dict[str, float]:
        """
        Advanced intent classification using multiple methods and dialogue context, with updated classes.
        
        Args:
            utterance: The user's message.
            user_id: The user's identifier.
            
        Returns:
            Dictionary of intent names and confidence scores.
        """
        if not utterance.strip():
            return {"out_of_scope": 1.0}
        
        # Add this message to conversation history
        self.add_to_conversation_history(user_id, utterance, "user")
        
        # Check cache first
        cache_key = f"{user_id}:{utterance}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # Fast path for common greetings
        normalized_utterance = utterance.lower().strip().rstrip('.!?')
        common_greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if normalized_utterance in common_greetings or normalized_utterance.startswith("hello ") or normalized_utterance.startswith("hi "):
            logger.info(f"Fast path: Detected greeting message: '{utterance}'")
            result = {"greeting": 0.95, "smallTalk": 0.05}
            self.classification_cache[cache_key] = result
            return result
        
        # First, check if this is an out-of-scope message
        out_of_scope_confidence = self.detect_out_of_scope(utterance)
        if out_of_scope_confidence > 0.7:  # High confidence threshold for out-of-scope
            logger.info(f"Detected out-of-scope message with confidence {out_of_scope_confidence:.2f}: '{utterance}'")
            result = {"out_of_scope": out_of_scope_confidence}
            self.classification_cache[cache_key] = result
            return result
        
        # For non-greeting messages, ensure LLM services are initialized
        await self._ensure_initialized()
        
        # Get dialogue context
        dialogue_state = self.dialogue_tracker.get_state(user_id)
        previous_question = self.get_previous_question(user_id)
        
        # Run classification methods
        methods_results = []
        
        # Method 1: Classify with dialogue context
        try:
            context_result = await self.classify_with_dialogue_context(utterance, user_id)
            methods_results.append(("dialogue_context", context_result))
        except Exception as e:
            logger.warning(f"Error in dialogue context classification: {str(e)}")
        
        # Method 2: Classify with medical knowledge
        try:
            medical_result = await self.classify_with_medical_knowledge(utterance)
            methods_results.append(("medical_knowledge", medical_result))
        except Exception as e:
            logger.warning(f"Error in medical knowledge classification: {str(e)}")
        
        # Method 3: Check for implied symptoms based on context
        try:
            implied_score = await self.analyze_implied_symptoms(utterance, dialogue_state, previous_question)
            if implied_score > 0.5:  # Only include if reasonably confident
                methods_results.append(("implied_symptoms", {"symptomReporting": implied_score}))
        except Exception as e:
            logger.warning(f"Error in implied symptom analysis: {str(e)}")
        
        # Method 4: LLM fallback if available and needed
        if self.chat_service is not None and (len(methods_results) < 2 or any(r[1] and max(r[1].values()) < 0.7 for r in methods_results)):
            try:
                llm_result = await self.generate_llm_fallback(utterance, user_id)
                methods_results.append(("llm_fallback", llm_result))
            except Exception as e:
                logger.warning(f"Error in LLM fallback classification: {str(e)}")
        
        # Combine results using weighted ensemble
        if methods_results:
            result = await self.weighted_ensemble_classification(methods_results, utterance, user_id)
            
            # Add a small confidence for out-of-scope based on our detection
            # This ensures it's considered in the ensemble but doesn't override other strong signals
            if "out_of_scope" not in result or result["out_of_scope"] < out_of_scope_confidence:
                result["out_of_scope"] = out_of_scope_confidence
        else:
            # Fallback to a simple classification if all methods failed
            result = {"out_of_scope": 0.8}  # Default to out-of-scope with high confidence if all else fails
            logger.warning(f"All classification methods failed for: '{utterance}'")
        
        # Cache the result
        self.classification_cache[cache_key] = result
        
        return result
    
    def detect_out_of_scope(self, utterance: str) -> float:
        """
        Detect if a message is out of scope for a medical assistant by comparing with examples
        and checking for non-medical content.
        
        Args:
            utterance: The user's message
            
        Returns:
            Confidence score for out-of-scope detection (0.0-1.0)
        """
        # Normalize the utterance
        normalized_utterance = utterance.lower().strip()
        
        # Direct match with examples (high confidence)
        for example in self.out_of_scope_examples:
            if normalized_utterance in example.lower() or example.lower() in normalized_utterance:
                return 0.95
        
        # Check for common non-medical topics
        non_medical_topics = [
            "weather", "restaurant", "food", "movie", "film", "tv", "show", "news", 
            "politics", "sports", "game", "joke", "funny", "computer", "technology", 
            "homework", "school", "university", "travel", "vacation", "holiday", 
            "music", "song", "book", "novel", "celebrity", "actor", "actress"
        ]
        
        # Count how many non-medical topics are mentioned
        topic_matches = sum(1 for topic in non_medical_topics if topic in normalized_utterance)
        
        # Calculate confidence based on topic matches
        if topic_matches >= 2:
            return 0.9
        elif topic_matches == 1:
            return 0.7
        
        # Check for question patterns about non-medical topics
        question_patterns = [
            "what is", "what's", "what are", "who is", "who's", "where is", 
            "where's", "when is", "when's", "how do", "how can", "can you tell me", 
            "do you know", "tell me about", "explain"
        ]
        
        if any(pattern in normalized_utterance for pattern in question_patterns):
            # If it's a question but doesn't contain medical terms, likely out of scope
            medical_terms = ["symptom", "pain", "doctor", "hospital", "medicine", "medical", 
                            "health", "disease", "condition", "treatment", "diagnosis", "sick", 
                            "feel", "hurt", "ache", "fever", "cough", "headache"]
            
            if not any(term in normalized_utterance for term in medical_terms):
                return 0.8
        
        # Default - low confidence in being out of scope
        return 0.1
