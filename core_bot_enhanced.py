import logging
import re
import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple

# Import our enhanced function calling modules
from llm_function_handler import LLMFunctionHandler
from diagnostic_engine_enhanced import DiagnosticEngine
from dialog_manager import DialogManager
from medical_knowledge_plugin import MedicalKnowledgePlugin
from intent_classification import IntentClassificationService

logger = logging.getLogger(__name__)

# Helper functions
def current_utc_timestamp() -> str:
    """Generate a UTC timestamp string."""
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def initialize_patient_data_if_needed(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the patient_data structure exists in user_data."""
    if "patient_data" not in user_data:
        user_data["patient_data"] = {
            "patient_id": "",
            "demographics": {
                "age": 0,
                "gender": "",
                "weight": 0.0,
                "height": 0.0,
                "other_demographics": {}
            },
            "symptoms": [],
            "medical_history": [],
            "asked_questions": [],
            "diagnosis": {
                "name": None,
                "confidence": 0.0
            },
            "mitigations": []
        }
    return user_data["patient_data"]

class ChatMessage:
    """Simple message class for chat history."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.timestamp = current_utc_timestamp()

class ChatHistory:
    """Maintains a history of chat messages."""
    def __init__(self, max_messages: int = 50):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to history."""
        self.messages.append(ChatMessage("user", message))
        self._trim_if_needed()
    
    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to history."""
        self.messages.append(ChatMessage("assistant", message))
        self._trim_if_needed()
    
    def _trim_if_needed(self) -> None:
        """Ensure history doesn't exceed maximum length."""
        if len(self.messages) > self.max_messages:
            # Keep the most recent messages
            self.messages = self.messages[-self.max_messages:]

class MedicalAssistantBot:
    """Enhanced medical assistant that uses function calling for medical tasks."""
    
    def __init__(self):
        """Initialize the medical assistant bot with function calling capabilities."""
        # Initialize enhanced components
        self.llm_handler = LLMFunctionHandler()
        self.medical_plugin = MedicalKnowledgePlugin()
        self.dialog_manager = DialogManager()
        self.diagnostic_engine = DiagnosticEngine(self.llm_handler, self.medical_plugin)
        
        # Initialize intent classifier with its own dedicated service
        self.intent_classifier = IntentClassificationService()
        
        # Chat histories by user ID
        self.chat_histories: Dict[str, ChatHistory] = {}
        
        # User data storage - includes 'patient_data' as well as other user-specific info
        self.user_data: Dict[str, Dict[str, Any]] = {}
        
        # Track model usage for responses
        self.model_usage: Dict[str, Dict[str, str]] = {}
    
    def get_chat_history(self, user_id: str) -> ChatHistory:
        """Get or create chat history for a user."""
        if user_id not in self.chat_histories:
            self.chat_histories[user_id] = ChatHistory()
        return self.chat_histories[user_id]
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get or create user data."""
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        return self.user_data[user_id]
    
    def track_model_usage(self, user_id: str, model_info: Dict[str, str]) -> None:
        """Track model usage for a user."""
        if user_id not in self.model_usage:
            self.model_usage[user_id] = {}
        
        self.model_usage[user_id] = model_info
    
    def get_model_usage(self, user_id: str) -> Dict[str, str]:
        """Get model usage for a user."""
        return self.model_usage.get(user_id, {"model": "unknown", "deployment": "unknown"})
    
    def reset_session(self, user_id: str) -> None:
        """
        Reset a user's session completely, clearing all data and history.
        This should be called when a new conversation begins after a farewell or diagnosis.
        
        Args:
            user_id: The user's identifier
        """
        logger.info(f"Resetting session for user {user_id}")
        
        # Reset user state in dialog manager
        self.dialog_manager.reset_user_state(user_id)
        
        # Clear chat history
        if user_id in self.chat_histories:
            self.chat_histories[user_id] = ChatHistory()
        
        # Clear user data
        if user_id in self.user_data:
            # Initialize a fresh patient data structure
            self.user_data[user_id] = {}
            initialize_patient_data_if_needed(self.user_data[user_id])
        
        logger.info(f"Session reset complete for user {user_id}")
    
    async def execute_action(self, action_name: str, user_id: str, user_message: str = "") -> str:
        """
        Execute a dialog action and return the response using function calling.
        
        Args:
            action_name: Name of the action to execute.
            user_id: User identifier.
            user_message: Original user message.
            
        Returns:
            Response text.
        """
        # Get user data and ensure patient_data is defined
        user_data = self.get_user_data(user_id)
        patient_data = initialize_patient_data_if_needed(user_data)
        
        logger.info(f"Executing action: {action_name}")
        
        if action_name == "utter_greet":
            return "Hello! I'm your friendly medical assistant. I'm here to help you with any health concerns you might be experiencing. How are you feeling today?"
        
        elif action_name == "utter_how_can_i_help":
            return "Have you been experiencing any specific symptoms or health concerns that I can assist you with? Please feel free to share as much or as little as you're comfortable with."
        
        elif action_name == "action_handle_out_of_scope":
            # Get the original message for context
            original_message = self.dialog_manager.get_original_message(user_id)
            
            try:
                # Use function calling for a consistent response
                result = await self.llm_handler.execute_chat_prompt(
                    f"The user has sent a message that appears to be outside the scope of a medical conversation. "
                    f"Their message was: \"{original_message}\". "
                    f"Provide a polite, brief response that acknowledges their message but gently redirects "
                    f"the conversation to medical topics. Make sure your response is concise (max 2 sentences) "
                    f"and ends with a question about their health concerns."
                )
                
                self.track_model_usage(user_id, {
                    "model": result.get("model", "unknown"),
                    "service_id": result.get("service_id", "unknown")
                })
                return result.get("text", "")
            except Exception as e:
                logger.error(f"Error handling out of scope message: {str(e)}")
                return f"I understand you're asking about \"{original_message}\", but I'm primarily designed to help with medical questions."
        
        elif action_name == "utter_redirect_to_medical":
            return "Is there something about your health I can help with today?"
        
        elif action_name == "action_ask_followup_question":
            # Generate a follow-up question using function calling
            response = await self.diagnostic_engine.generate_followup_question(patient_data)
            
            # After asking a follow-up, update the diagnosis confidence
            await self.diagnostic_engine.update_diagnosis_confidence(patient_data)
            return response
        
        elif action_name == "action_verify_symptoms":
            # Use function calling to verify symptoms
            verification_response = await self.diagnostic_engine.verify_symptoms(patient_data)
            return verification_response
        
        elif action_name == "action_provide_diagnosis":
            # Generate diagnosis using function calling
            diagnosis_response = await self.diagnostic_engine.generate_diagnosis(patient_data)
            return diagnosis_response
        
        elif action_name == "utter_suggest_mitigations":
            # Generate mitigation suggestions using function calling
            mitigations_response = await self.diagnostic_engine.suggest_mitigations(patient_data)
            return mitigations_response
        
        elif action_name == "utter_emergency_response":
            # Handle emergency situations
            return "MEDICAL EMERGENCY DETECTED: Please seek immediate medical attention by calling emergency services or going to the nearest emergency room. Do not wait."
        
        elif action_name == "utter_farewell":
            return "Thank you for using the medical assistant. Take care and don't hesitate to return if you have more health concerns."
        
        else:
            logger.warning(f"Unknown action: {action_name}")
            return "I'm not sure how to respond to that. Could you please rephrase or ask me about your medical concerns?"
    
    async def process_message(self, user_id: str, message: str, include_diagnostics: bool = False) -> str:
        """
        Process a user message and return a response.
        Enhanced with function calling for more structured and consistent responses.
        
        Args:
            user_id: The user's identifier
            message: The user's message
            include_diagnostics: Whether to include diagnostic information in the response
            
        Returns:
            Bot response
        """
        # Get user history and data
        history = self.get_chat_history(user_id)
        user_data = self.get_user_data(user_id)
        
        # Add user message to history
        history.add_user_message(message)
        
        # Generate user context from history
        user_context = await self._generate_user_context(history)
        
        # Classify the intent of the user's message using function calling
        try:
            # Get intent classification
            intent_result = await self.intent_classifier.classify_intent(message, user_context["text"])
            top_intent = intent_result["intent"]
            top_score = intent_result["confidence"]
            
            logger.info(f"Classified intent: {top_intent} with confidence {top_score:.2f}")
            
            # Store the original message and intent in dialog manager for reference
            self.dialog_manager.set_original_message(user_id, message)
            self.dialog_manager.set_message_intent(user_id, top_intent)
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}")
            top_intent = "medicalInquiry"  # Default fallback
            top_score = 0.6
        
        # Ensure patient_data is initialized
        patient_data = initialize_patient_data_if_needed(user_data)
        
        # Process symptoms if intent is symptom reporting
        if top_intent == "symptomReporting" and message:
            # Get the current dialog state
            current_state = self.dialog_manager.get_user_state(user_id)
            
            # If we already have a diagnosis and user reports new symptoms,
            # we should reset the session and start fresh
            if (current_state in ["generating_diagnosis", "verification"] and 
                patient_data.get("diagnosis", {}).get("name")):
                logger.info("New symptoms reported after diagnosis - resetting session")
                self.reset_session(user_id)
                # Get fresh references after reset
                history = self.get_chat_history(user_id)
                user_data = self.get_user_data(user_id)
                patient_data = initialize_patient_data_if_needed(user_data)
                history.add_user_message(message)
            
            # Use function calling to extract symptoms from message
            symptoms = await self.diagnostic_engine.extract_symptoms_from_message(message)
            
            # Add each extracted symptom
            for symptom_text in symptoms:
                logger.info(f"Adding symptom: '{symptom_text}' to patient data")
                self.diagnostic_engine.add_symptom(patient_data, symptom_text)
            
            # Update diagnosis confidence based on new symptoms
            await self.diagnostic_engine.update_diagnosis_confidence(patient_data)
            logger.info(f"After adding symptoms, patient data: {patient_data.get('symptoms', [])}")
            logger.info(f"Diagnosis confidence updated to: {patient_data.get('diagnosis', {}).get('confidence', 0.0)}")
            
            # Check for emergency situations
            emergency_result = await self.diagnostic_engine.detect_emergency(patient_data)
            if emergency_result.get("is_emergency", False):
                # If emergency is detected, immediately respond with emergency message
                emergency_message = emergency_result.get("emergency_message")
                logger.warning(f"EMERGENCY DETECTED - Responding with: {emergency_message}")
                
                # Add bot message to history
                history.add_assistant_message(emergency_message)
                
                # Set a special emergency state
                self.dialog_manager.set_user_state(user_id, "emergency")
                
                # Return the emergency message immediately
                return emergency_message
            
            # Possibly move to verification if confidence is high enough
            if self.dialog_manager.should_verify_symptoms(user_id, patient_data):
                logger.info("Confidence threshold reached, transitioning to verification")
        
        # Check if we need to immediately generate a diagnosis report based on O1 verification
        if patient_data.get("ready_for_report", False) and patient_data.get("verification_complete", False):
            logger.info("O1 model verified diagnosis with high confidence, generating medical report")
            # Force transition to diagnosis generation state
            next_state = "generating_diagnosis"
            # Clear the flags to prevent repeated generation
            patient_data["ready_for_report"] = False
            patient_data["verification_complete"] = False
            
            # Generate the diagnosis and mitigations using function calling
            diagnosis_response = await self.diagnostic_engine.generate_diagnosis(patient_data)
            mitigations_response = await self.diagnostic_engine.suggest_mitigations(patient_data)
            
            # Combine responses
            responses = [diagnosis_response, mitigations_response]
            
            # Add a message asking if there's anything else the user needs help with
            responses.append("Is there anything else I can help you with today?")
            
            # Update user state to generating_diagnosis
            self.dialog_manager.set_user_state(user_id, next_state)
        else:
            # Standard dialog flow handling
            # Determine next state based on current state and intent
            next_state = self.dialog_manager.advance_dialog_state(user_id, top_intent)
            
            # Get the next actions to execute
            next_actions = self.dialog_manager.get_next_actions(next_state)
            
            # Execute actions and collect responses
            responses = []
            for action in next_actions:
                response = await self.execute_action(action, user_id, message)
                responses.append(response)
            
            # Update user state
            self.dialog_manager.set_user_state(user_id, next_state)
        
        # Combine responses
        full_response = " ".join(responses)
        
        # Add assistant response to history
        history.add_assistant_message(full_response)
        
        # Get model information
        model_info = self.get_model_usage(user_id)
        
        # Optionally include diagnostic information
        if include_diagnostics:
            intent_details = f"{top_intent} (confidence: {top_score:.2f})"
            diagnostic_info = self._generate_diagnostic_info(
                user_id, patient_data, intent_details, user_context, model_info
            )
            enhanced_response = f"{full_response}\n\n{diagnostic_info}"
            return enhanced_response
        else:
            return full_response
    
    async def _generate_user_context(self, history: ChatHistory) -> Dict[str, Any]:
        """
        Generate a summarized user context from conversation history using function calling.
        
        Args:
            history: The chat history.
                
        Returns:
            Dictionary with context text and model info.
        """
        if len(history.messages) < 3:
            return {
                "text": "No context generated",
                "model": "none",
                "service_id": "none"
            }
        
        try:
            # Convert history to text format
            history_text = ""
            for msg in history.messages:
                role = "User" if msg.role.lower() == "user" else "Assistant"
                history_text += f"{role}: {msg.content}\n"
                
            # Execute chat prompt for context generation
            result = await self.llm_handler.execute_chat_prompt(
                f"Based on the following conversation history, create a brief summary of the user's context, "
                f"including key health concerns, symptoms mentioned, and relevant details.\n\n"
                f"CONVERSATION HISTORY:\n{history_text}\n\n"
                f"Provide a concise (2-3 sentence) summary of this user's context."
            )
            
            return result
        except Exception as e:
            logger.error(f"Error generating user context: {str(e)}")
            return {
                "text": "Context generation failed",
                "model": "error",
                "service_id": "none"
            }
    
    def _generate_diagnostic_info(self, user_id: str, patient_data: Dict[str, Any], 
                                intent: str, context: Dict[str, Any], 
                                model_info: Dict[str, str]) -> str:
        """
        Generate diagnostic information for debugging.
        
        Args:
            user_id: The user's identifier
            patient_data: Patient data
            intent: Classified intent
            context: Generated user context
            model_info: Model usage information
            
        Returns:
            Formatted diagnostic information
        """
        state = self.dialog_manager.get_user_state(user_id)
        symptoms = ", ".join(patient_data.get("symptoms", []))
        confidence = patient_data.get("diagnosis", {}).get("confidence", 0.0)
        diagnosis = patient_data.get("diagnosis", {}).get("name", "Unknown")
        
        diagnostic_str = f"DIAGNOSTIC INFO:\n"
        diagnostic_str += f"- User state: {state}\n"
        diagnostic_str += f"- Intent: {intent}\n"
        diagnostic_str += f"- User context: {context['text']}\n"
        diagnostic_str += f"- Symptoms: {symptoms}\n"
        diagnostic_str += f"- Diagnosis: {diagnosis} (confidence: {confidence:.2f})\n"
        diagnostic_str += f"- Model: {model_info.get('model', 'unknown')}\n"
        
        return diagnostic_str

# External functions for API and CLI interfaces
_bot_instances = {}

async def process_message_api(message: str, user_id: str = None, include_diagnostics: bool = False) -> str:
    """
    Process a single message from an API request and return the response with diagnostic information.
    
    Args:
        message: The user's message text
        user_id: Optional user ID to maintain conversation state between requests
        include_diagnostics: Whether to include diagnostic info in the response
        
    Returns:
        The bot's response as a string with optional diagnostic information
    """
    global _bot_instances
    
    if not user_id:
        user_id = f"api_user_{hash(message) % 10000}"
    
    if user_id not in _bot_instances:
        logger.info(f"Creating new bot instance for user: {user_id}")
        _bot_instances[user_id] = MedicalAssistantBot()
    
    bot = _bot_instances[user_id]
    
    try:
        response = await bot.process_message(user_id, message, include_diagnostics=include_diagnostics)
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing API message for user {user_id}: {str(e)}\n{error_details}")
        return "I'm sorry, I encountered an error processing your message. Please try again."

async def interactive_conversation():
    """Run an interactive conversation with the medical assistant bot with enhanced debugging."""
    import logging
    import traceback
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    bot = MedicalAssistantBot()
    user_id = f"cli_user_{random.randint(1000, 9999)}"
    
    print("\nMedical Assistant Bot (using function calling)")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'debug on' to enable diagnostic info, 'debug off' to disable.")
    
    debug_mode = False
    
    # Initial greeting
    initial_response = await bot.execute_action("utter_greet", user_id)
    print(f"\nBot: {initial_response}")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\nBot: Goodbye! Take care.")
                break
            
            # Check for debug commands
            elif user_input.lower() == "debug on":
                debug_mode = True
                print("\nBot: Debug mode enabled.")
                continue
            elif user_input.lower() == "debug off":
                debug_mode = False
                print("\nBot: Debug mode disabled.")
                continue
            
            # Process the message
            response = await bot.process_message(user_id, user_input, include_diagnostics=debug_mode)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nExiting conversation. Goodbye!")
            break
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"\nError: {str(e)}")
            print(f"Details: {error_details}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(interactive_conversation())
