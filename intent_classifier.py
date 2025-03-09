"""
Advanced Medical Intent Classification System with Dialogue State Tracking
"""
import os
import re
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Set
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
    """Tracks the state of the dialogue to provide context for intent classification"""
    
    def __init__(self):
        """Initialize the dialogue state tracker"""
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
        """Get the current dialogue state for a user"""
        return self.user_states.get(user_id, "greeting")
    
    def update_state(self, user_id: str, state: str) -> None:
        """Update the dialogue state for a user"""
        if state in self.states:
            self.user_states[user_id] = state
    
    def set_expected_response(self, user_id: str, expected_type: str) -> None:
        """
        Set the type of response expected from the user
        
        Args:
            user_id: The user's identifier
            expected_type: Type of expected response (e.g., "yes_no", "symptom_details")
        """
        self.expected_responses[user_id] = expected_type
    
    def get_expected_response(self, user_id: str) -> Optional[str]:
        """Get the type of response expected from the user"""
        return self.expected_responses.get(user_id)
    
    def update_from_assistant_message(self, user_id: str, message: str) -> None:
        """
        Update dialogue state based on assistant's message
        
        Args:
            user_id: The user's identifier
            message: The assistant's message
        """
        message_lower = message.lower()
        
        # Check if the message is asking about symptoms
        if any(phrase in message_lower for phrase in [
            "what symptoms", "any symptoms", "how are you feeling", 
            "tell me about your", "experiencing any", "describe your",
            "having any other", "when did", "how long"
        ]):
            self.update_state(user_id, "asking_followup")
            self.set_expected_response(user_id, "symptom_details")
        
        # Check if the message is asking for confirmation
        elif "?" in message and any(phrase in message_lower for phrase in [
            "is that correct", "is this right", "do you", "have you", 
            "would you", "could you", "are you"
        ]):
            self.set_expected_response(user_id, "yes_no")
            
            # If we're in verification mode
            if "understand you're experiencing" in message_lower:
                self.update_state(user_id, "verification")
        
        # Check if providing information
        elif any(phrase in message_lower for phrase in [
            "here's some information", "let me explain", "let me tell you about",
            "is a condition", "treatment", "causes", "symptoms of"
        ]):
            self.update_state(user_id, "providing_info")
            self.set_expected_response(user_id, "acknowledgment")
        
        # Check if closing
        elif any(phrase in message_lower for phrase in [
            "take care", "goodbye", "hope you feel better", "anything else",
            "anything else i can help with"
        ]):
            self.update_state(user_id, "closing")
            self.set_expected_response(user_id, "farewell")


class MedicalIntentClassifier:
    """Advanced intent classification with contextual understanding for medical dialogues"""
    
    def __init__(self, chat_service: AzureChatCompletion = None, kernel: Kernel = None):
        """
        Initialize the intent classifier
        
        Args:
            chat_service: Azure Chat Completion service for LLM-based classification (optional)
            kernel: Semantic Kernel instance for plugin access (optional)
        """
        # Load environment variables
        load_dotenv()
        
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
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
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
    
    def add_to_conversation_history(self, user_id: str, message: str, role: str = "user") -> None:
        """
        Add a message to the conversation history for a user
        
        Args:
            user_id: The user's identifier
            message: The message text
            role: The role of the message sender ("user" or "assistant")
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
            self.dialogue_tracker.update_from_assistant_message(user_id, message)
            
            # If it's a question, store it to help with follow-up classification
            if "?" in message:
                self.previous_questions[user_id] = message
    
    def get_formatted_history(self, user_id: str, max_turns: int = 3) -> str:
        """
        Get formatted conversation history for a user
        
        Args:
            user_id: The user's identifier
            max_turns: Maximum number of conversation turns to include
            
        Returns:
            Formatted conversation history
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
        """Get the most recent question from the assistant"""
        return self.previous_questions.get(user_id, "")
    
    async def analyze_implied_symptoms(self, utterance: str, dialogue_state: str, previous_question: str) -> Optional[float]:
        """
        Analyze if a message contains implied symptoms based on dialogue context
        
        Args:
            utterance: The user's message
            dialogue_state: Current dialogue state
            previous_question: Previous question from the assistant
            
        Returns:
            Confidence score for implied symptom detection
        """
        if not self.chat_service:
            return None
        
        # Skip this analysis for certain message types
        if len(utterance.split()) <= 1:  # Very short responses
            return None
        
        prompt = f"""You're analyzing a patient's response to determine if it contains health-related information, even if implicit.

Previous assistant question: "{previous_question}"
Current dialogue state: {dialogue_state}
Patient's response: "{utterance}"

In a medical conversation, patients often provide symptom information indirectly or implicitly. 
Consider all possible ways this response might contain or imply health information:

1. Direct symptom descriptions
2. Descriptions of bodily sensations or experiences
3. Timeframes that relate to a health condition
4. Information about activities affected by health
5. Emotional states that could be health-related
6. Descriptions of changes in physical or mental state
7. Responses to questions about health that confirm or deny
8. Information about medications or treatments

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
        Classify intent with full dialogue context awareness
        
        Args:
            utterance: The user's message
            user_id: The user's identifier
            
        Returns:
            Dictionary of intent names and confidence scores
        """
        if not self.chat_service:
            return None
        
        # Get dialogue context
        dialogue_state = self.dialogue_tracker.get_state(user_id)
        expected_response = self.dialogue_tracker.get_expected_response(user_id)
        conversation_history = self.get_formatted_history(user_id)
        previous_question = self.get_previous_question(user_id)
        
        # Create a contextual prompt
        prompt = f"""You are a medical assistant analyzing patient intent with full dialogue context.

CONVERSATION HISTORY:
{conversation_history}

DIALOGUE CONTEXT:
- Current dialogue state: {dialogue_state}
- Expected response type: {expected_response or "None"}
- Previous assistant question: "{previous_question}"

CURRENT USER MESSAGE: "{utterance}"

Given this full context, classify the user's intent into one of these categories:

- greet: Initial greeting or introduction
- inform_symptoms: ANY statement describing health concerns, symptoms, physical or mental health issues - including indirect or implicit references
- ask_medical_info: Questions about health topics or medical information
- confirm: Affirmative responses (yes, that's right, etc.)
- deny: Negative responses (no, that's not right, etc.)
- goodbye: Closing the conversation
- out_of_scope: Unrelated to medical discussion

CONTEXT-AWARE ANALYSIS GUIDANCE:
- If the assistant just asked about symptoms and the user responds with ANY information, it's likely "inform_symptoms"
- Short answers to medical questions often contain implicit symptom information
- Responses to questions that ask "do you have X?" are likely confirmations/denials AND symptom information
- Even responses like "for about a week" to duration questions are "inform_symptoms"
- In the "collecting_symptoms" or "asking_followup" states, most responses will be "inform_symptoms" unless clearly different
- Pay special attention to the expected response type and dialogue state

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
        Classify intent with focus on medical terminology and health expressions
        
        Args:
            utterance: The user's message
            
        Returns:
            Dictionary of intent names and confidence scores
        """
        if not self.chat_service:
            return None
        
        prompt = f"""As a medical professional, analyze this patient message to determine its intent:

Patient message: "{utterance}"

Medical professionals understand that patients describe health concerns in many ways, from medical terminology to colloquial expressions.
Patients may use vague terms, describe impacts rather than symptoms directly, or mention duration/patterns.

Classify this message into one of these intent categories:

- greet: Initial greeting or introduction
- inform_symptoms: ANY statement describing health experiences, sensations, concerns, or medical conditions
- ask_medical_info: Questions about health topics or medical information
- confirm: Affirmative responses (yes, that's right, etc.)
- deny: Negative responses (no, that's not right, etc.)
- goodbye: Closing the conversation
- out_of_scope: Unrelated to medical discussion

MEDICAL INTENT ANALYSIS:
- "Inform_symptoms" covers ALL expressions of physical/mental experiences or health concerns
- Consider how patients in different demographics express health concerns
- Implicit expressions like "I haven't been sleeping well" count as symptom information
- Mentions of medication effects or changes in condition are symptom information
- Health timeline information is symptom information
- In a medical context, even vague expressions of unwellness should be classified as symptom information

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
    
    def _create_fallback_intent_scores(self, utterance: str, user_id: str) -> Dict[str, float]:
        """
        Create a context-aware fallback intent classification when LLM calls fail
        
        Args:
            utterance: The user's message
            user_id: The user's identifier
            
        Returns:
            Dictionary of intent scores based on dialogue context
        """
        # Set default scores with higher base score for symptom intent in medical context
        scores = {intent: 0.1 for intent in self.intent_options}
        
        # Get dialogue context to influence the fallback
        dialogue_state = self.dialogue_tracker.get_state(user_id)
        expected_response = self.dialogue_tracker.get_expected_response(user_id)
        
        # Adjust base scores based on dialogue state
        if dialogue_state in ["collecting_symptoms", "asking_followup"]:
            # In these states, messages are more likely to be symptom information
            scores["inform_symptoms"] = 0.5
        elif dialogue_state == "verification":
            # In verification state, responses are likely confirms or denies
            scores["confirm"] = 0.4
            scores["deny"] = 0.4
        elif dialogue_state == "closing":
            # In closing state, responses are likely goodbyes or confirms
            scores["goodbye"] = 0.4
            scores["confirm"] = 0.3
        else:
            # Default higher chance of symptoms in a medical context
            scores["inform_symptoms"] = 0.3
        
        # Adjust based on expected response type
        if expected_response == "symptom_details":
            scores["inform_symptoms"] = 0.6
        elif expected_response == "yes_no":
            scores["confirm"] = 0.4
            scores["deny"] = 0.4
        
        # Simple pattern matching for critical intents
        utterance_lower = utterance.lower()
        
        # Check for yes/no responses
        if any(word in utterance_lower for word in ["yes", "yeah", "correct", "right", "sure"]):
            scores["confirm"] = 0.8
            # If confirming in collecting_symptoms or verification state, it's also symptom info
            if dialogue_state in ["collecting_symptoms", "verification", "asking_followup"]:
                scores["inform_symptoms"] = 0.7
            else:
                scores["inform_symptoms"] = 0.2
        elif any(word in utterance_lower for word in ["no", "not ", "don't", "nope", "incorrect"]):
            scores["deny"] = 0.8
            # If denying in collecting_symptoms or verification state, it's also symptom info
            if dialogue_state in ["collecting_symptoms", "verification", "asking_followup"]:
                scores["inform_symptoms"] = 0.7
            else:
                scores["inform_symptoms"] = 0.2
        
        # Check for greeting patterns
        elif any(greeting in utterance_lower for greeting in ["hello", "hi ", "hey", "greetings"]):
            scores["greet"] = 0.8
            scores["inform_symptoms"] = 0.1
        
        # Check for goodbye patterns
        elif any(word in utterance_lower for word in ["bye", "goodbye", "thanks", "thank you"]):
            scores["goodbye"] = 0.8
            scores["inform_symptoms"] = 0.1
        
        # Check for questions (might be medical info requests)
        elif "?" in utterance or utterance_lower.startswith(("what", "how", "when", "where", "why", "can", "could")):
            scores["ask_medical_info"] = 0.7
            scores["inform_symptoms"] = 0.2
        
        return scores
    
    async def weighted_ensemble_classification(self, methods_results: List[Tuple[str, Dict[str, float]]], 
                                              utterance: str, user_id: str) -> Dict[str, float]:
        """
        Combine results from multiple classification methods using a dynamic weighted ensemble
        
        Args:
            methods_results: List of (method_name, result) tuples from different methods
            utterance: The user's message
            user_id: The user's identifier
            
        Returns:
            Combined intent scores
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
            # For simple interactions, all methods are roughly equal
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
                
                # Special handling for implied symptoms analysis
                if method_name == "implied_symptoms":
                    # This is just a confidence score for "inform_symptoms"
                    confidence = result
                    if confidence > 0.5:
                        combined_scores["inform_symptoms"] += confidence * method_weight
                        total_weight += method_weight
                else:
                    # Regular classification result
                    for intent, score in result.items():
                        combined_scores[intent] += score * method_weight
                    
                    total_weight += method_weight
        
        # If no methods returned results, return the fallback
        if total_weight < 0.1:
            logger.warning("No classification methods succeeded, using fallback")
            return self._create_fallback_intent_scores(utterance, user_id)
        
        # Normalize scores based on applied weight
        for intent in combined_scores:
            combined_scores[intent] = combined_scores[intent] / total_weight
        
        # Get the top intent and score
        top_intent = max(combined_scores.items(), key=lambda x: x[1])[0]
        top_score = combined_scores[top_intent]
        
        logger.info(f"Ensemble classification result: {top_intent} ({top_score:.2f})")
        return combined_scores
    
    async def classify_intent(self, utterance: str, user_id: str = "default_user") -> Dict[str, float]:
        """
        Advanced intent classification using multiple methods and dialogue context
        
        Args:
            utterance: The user's message
            user_id: The user's identifier
            
        Returns:
            Dictionary of intent names and confidence scores
        """
        if not utterance.strip():
            return {"out_of_scope": 1.0}
        
        # Add this message to conversation history
        self.add_to_conversation_history(user_id, utterance, "user")
        
        # Get dialogue context
        dialogue_state = self.dialogue_tracker.get_state(user_id)
        previous_question = self.get_previous_question(user_id)
        
        # Check cache first for very frequent identical messages
        cache_key = f"{user_id}:{utterance}:{dialogue_state}"
        if cache_key in self.classification_cache:
            logger.info(f"Using cached classification result for: {utterance}")
            return self.classification_cache[cache_key]
        
        # Run classification methods in parallel
        tasks = [
            self.classify_with_dialogue_context(utterance, user_id),
            self.classify_with_medical_knowledge(utterance),
            self.analyze_implied_symptoms(utterance, dialogue_state, previous_question)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, filtering out exceptions
        valid_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Classification method error: {str(result)}")
            elif result is not None:
                method_name = ["dialogue_context", "medical_knowledge", "implied_symptoms"][i]
                valid_results.append((method_name, result))
        
        # Combine results with weighted ensemble
        if valid_results:
            ensemble_result = await self.weighted_ensemble_classification(valid_results, utterance, user_id)
            
            # Update cache with result
            self.classification_cache[cache_key] = ensemble_result
            
            return ensemble_result
        
        # If all methods failed, use the fallback
        logger.warning("All classification methods failed, using dialogue-aware fallback")
        fallback_result = self._create_fallback_intent_scores(utterance, user_id)
        
        # Cache the fallback result too
        self.classification_cache[cache_key] = fallback_result
        
        return fallback_result