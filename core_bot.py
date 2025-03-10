"""
Core medical assistant bot using Semantic Kernel

This is the main module that coordinates all components of the medical assistant bot.
It handles message processing, orchestrates dialog flow, and integrates the diagnostic
engine with self-reflection based confidence calculation.

Changes made:
- Removed fallback checks for LLM availability and any "last resort fallback" logic.
- Introduced a standardized patient data model under user_data["patient_data"].
- Added a simple way to parse demographic info (age, gender, weight, height) from user messages and update the patient_data model accordingly.
- Kept all other functionality intact as requested.
"""
import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import time

# Semantic Kernel imports
from semantic_kernel.contents.chat_history import ChatHistory

# Local imports - our modular components
from dialog_manager import DialogManager
from llm_handler import LLMHandler
from diagnostic_engine import DiagnosticEngine
from medical_plugins import MedicalKnowledgePlugin
from intent_classifier import MedicalIntentClassifier as IntentClassificationService

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def initialize_patient_data_if_needed(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure that user_data has a 'patient_data' key with the standard structure.
    Returns the patient_data dict.
    """
    if "patient_data" not in user_data:
        user_data["patient_data"] = {
            "patient_id": "",  # can be set to a real ID or remain empty
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

def parse_and_update_demographics(patient_data: Dict[str, Any], message: str) -> None:
    """
    Very simple demonstration parser that checks the user message
    for lines like 'age: 40', 'gender: male', 'weight: 72.5', 'height: 170.2'
    and updates the patient_data accordingly.
    """
    # For example, look for lines 'age: <number>' etc.
    # This is a naive approach; in a real system, you'd use a more robust parser.
    lines = message.split("\n")
    for line in lines:
        parts = line.lower().split(":")
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            if key == "age":
                try:
                    patient_data["demographics"]["age"] = int(value)
                except ValueError:
                    pass
            elif key == "gender":
                patient_data["demographics"]["gender"] = value
            elif key == "weight":
                try:
                    patient_data["demographics"]["weight"] = float(value)
                except ValueError:
                    pass
            elif key == "height":
                try:
                    patient_data["demographics"]["height"] = float(value)
                except ValueError:
                    pass
    # Additional logic for other demographics can go here if needed.

class MedicalAssistantBot:
    """Flexible medical assistant that can handle any medical issue."""
    
    def __init__(self):
        """Initialize the medical assistant bot."""
        # Initialize components
        self.llm_handler = LLMHandler()
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
    
    async def execute_action(self, action_name: str, user_id: str, user_message: str = "") -> str:
        """
        Execute a dialog action and return the response.
        
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
            return "Hello! I'm your medical assistant. I'm here to help with your health questions."
        
        elif action_name == "utter_how_can_i_help":
            return "How can I help you today?"
        
        elif action_name == "action_handle_out_of_scope":
            # Get the original message for context
            original_message = self.dialog_manager.get_original_message(user_id)
            
            # Removed fallback checks; always attempt LLM
            try:
                prompt = f"""The user has sent a message that appears to be outside the scope of a medical conversation. 
Their message was: "{original_message}"

Provide a polite, brief response that acknowledges their message but gently redirects the conversation to medical topics.
Make sure your response is concise (max 2 sentences) and ends with a question about their health concerns."""
                
                response_data = await self.llm_handler.execute_prompt(prompt)
                self.track_model_usage(user_id, {
                    "model": response_data.get("model", "unknown"),
                    "deployment": response_data.get("deployment", "unknown")
                })
                return response_data.get("text", "")
            except Exception as e:
                logger.error(f"Error handling out of scope message with LLM: {str(e)}")
                return f"I understand you're asking about \"{original_message}\", but I'm primarily designed to help with medical questions."
        
        elif action_name == "utter_redirect_to_medical":
            return "Is there something about your health I can help with today?"
        
        elif action_name == "action_ask_followup_question":
            # Use the diagnostic engine to generate a follow-up question
            response = await self.diagnostic_engine.generate_followup_question(patient_data)
            
            # After asking a follow-up, update the diagnosis confidence
            await self.diagnostic_engine.update_diagnosis_confidence(patient_data)
            return response
        
        elif action_name == "action_verify_symptoms":
            # Verify symptoms before providing diagnosis
            return await self.diagnostic_engine.verify_symptoms(patient_data)
        
        elif action_name == "action_provide_medical_info":
            topic = user_message
            try:
                # Provide general medical information
                prompt = f"""Provide general medical information about the following topic:
Topic: {topic}
Patient demographics: {patient_data.get("demographics", {})}

Give helpful, accurate information while emphasizing this is general advice and not a substitute for professional medical care."""
                
                response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=True)
                self.track_model_usage(user_id, {
                    "model": response_data.get("model", "unknown"),
                    "deployment": response_data.get("deployment", "unknown")
                })
                return response_data.get("text", "")
            except Exception as e:
                logger.error(f"Error providing medical information with LLM: {str(e)}")
            
            # Attempt the plugin if an exception occurred
            if self.medical_plugin:
                try:
                    response = await self.medical_plugin.provide_medical_information(
                        topic=topic,
                        patient_demographics=str(patient_data.get("demographics", {}))
                    )
                    return str(response)
                except Exception as e:
                    logger.error(f"Error providing medical information via plugin: {str(e)}")
            
            return f"I can provide general information about {topic}, but remember to consult with a healthcare professional for personalized advice."
        
        elif action_name == "action_provide_diagnosis":
            # Use the diagnostic engine to generate a diagnosis
            return await self.diagnostic_engine.generate_diagnosis(patient_data)
        
        elif action_name == "utter_suggest_mitigations":
            # Use the diagnostic engine to suggest mitigations
            return await self.diagnostic_engine.suggest_mitigations(patient_data)
        
        elif action_name == "utter_anything_else":
            return "Is there anything else you'd like to know or discuss?"
        
        elif action_name == "utter_goodbye":
            return "Take care and don't hesitate to return if you have more questions. Goodbye!"
        
        else:
            logger.warning(f"Unknown action: {action_name}")
            return "I'm not sure how to respond to that."
    
    async def _generate_user_context(self, history: ChatHistory) -> Dict[str, Any]:
        """
        Generate a summarized user context from conversation history.
        
        Args:
            history: The chat history.
                
        Returns:
            Dictionary with context text and model info.
        """
        # Removed llm availability check; always try the prompt
        if len(history.messages) < 3:
            return {
                "text": "No context generated",
                "model": "none",
                "deployment": "none"
            }
        
        try:
            # Convert history to text format
            history_text = ""
            for msg in history.messages:
                role = "User" if msg.role.lower() == "user" else "Assistant"
                history_text += f"{role}: {msg.content}\n"
            
            # Create a prompt for context generation
            prompt = f"""Based on the following conversation history, create a brief summary of the user's context, 
including key health concerns, symptoms mentioned, and relevant details.

CONVERSATION HISTORY:
{history_text}

Provide a concise (2-3 sentence) summary of this user's context.
"""
            
            response_data = await self.llm_handler.execute_prompt(prompt, use_full_model=False)
            return response_data
        except Exception as e:
            logger.error(f"Error generating user context: {str(e)}")
            return {
                "text": "Context generation failed",
                "model": "error",
                "deployment": "none"
            }

    def _generate_diagnostic_info(self, user_id: str, patient_data: Dict[str, Any], intent: str, user_context: Dict[str, Any], model_info: Dict[str, str]) -> str:
        """
        Generate a formatted diagnostic information block.
        
        Args:
            user_id: The user's identifier
            patient_data: The patient data
            intent: The top intent
            user_context: User context data including text and model
            model_info: Model information for the main response
                
        Returns:
            Formatted diagnostic information
        """
        # Get current dialog state
        current_state = self.dialog_manager.get_user_state(user_id)
        
        # Get current date and time for the report
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get patient demographics
        demographics = patient_data.get("demographics", {})
        patient_name = demographics.get("name", "[Patient Name]")
        patient_age = demographics.get("age", "[Age]")
        patient_gender = demographics.get("gender", "[Gender]")
        patient_height = demographics.get("height", "[Height]")
        patient_weight = demographics.get("weight", "[Weight]")
        
        # Handle symptoms display - support both string and dict formats
        symptom_objs = patient_data.get("symptoms", [])
        if symptom_objs:
            symptom_texts = []
            for s in symptom_objs:
                if isinstance(s, dict):
                    symptom_text = s.get("name", "unknown")
                    if "additional_info" in s:
                        symptom_text += f" ({s['additional_info']})"
                    if "timestamp" in s:
                        symptom_text += f" - reported {s['timestamp']}"
                else:
                    symptom_text = str(s)
                symptom_texts.append(f"• {symptom_text}")
            symptoms_text = "\n".join(symptom_texts)
        else:
            symptoms_text = "No symptoms reported"
        
        # Get diagnosis information
        diagnosis_info = patient_data.get("diagnosis", {})
        diagnosis_name = diagnosis_info.get("name", "Pending diagnosis")
        confidence = diagnosis_info.get("confidence", 0.0)
        confidence_text = f"{confidence:.2f} ({confidence * 100:.1f}%)"
        
        # Additional reasoning or placeholders
        confidence_reasoning = patient_data.get("confidence_reasoning", "Assessment pending")
        
        # Get additional medical information
        asked_questions = patient_data.get("asked_questions", [])
        followup_count = len(asked_questions)
        
        # Format the questions as a medical assessment
        assessment_notes = []
        if asked_questions:
            for q in asked_questions:
                if isinstance(q, dict):
                    question_text = q.get("question", "Unknown question")
                    answer_text = q.get("answer", "")
                    if answer_text:
                        assessment_notes.append(f"• Q: {question_text}\n  A: {answer_text}")
                    else:
                        assessment_notes.append(f"• Q: {question_text}")
                else:
                    assessment_notes.append(f"• {q}")
            assessment_text = "\n".join(assessment_notes)
        else:
            assessment_text = "Initial assessment in progress"
        
        # Check for verification information
        verification_info = patient_data.get("verification_info", {})
        verification_trigger = verification_info.get("trigger_reason", "none")
        low_confidence_explanation = verification_info.get("low_confidence_explanation", "")
        referral_needed = patient_data.get("referral_needed", False)
        
        # Generate recommendations based on the diagnosis state and verification
        if verification_trigger == "low_confidence" and referral_needed:
            recommendations = "MEDICAL REFERRAL RECOMMENDED\n\n"
            if low_confidence_explanation:
                recommendations += low_confidence_explanation
            else:
                recommendations += "Based on the assessment, your symptoms require further evaluation by a medical professional.\n"
                recommendations += "• Please consult with a healthcare provider for a comprehensive evaluation\n"
                recommendations += "• Your symptoms do not clearly match a single condition with high confidence\n"
                recommendations += "• A medical professional can conduct additional tests and provide appropriate care"
        elif diagnosis_name != "Pending diagnosis" and confidence >= 0.6:
            recommendations = "Based on the assessment, the following is recommended:\n"
            recommendations += "• Consult with a healthcare professional for confirmation\n"
            recommendations += "• Monitor symptoms and seek immediate care if condition worsens\n"
            recommendations += "• Follow standard treatment protocols for the identified condition"
        else:
            recommendations = "Recommendations pending completion of assessment"
        
        # Extract model information for technical appendix
        context_model = f"{user_context.get('model', 'standard')}/{user_context.get('deployment', 'primary')}"
        response_model = f"{model_info.get('model', 'standard')}/{model_info.get('deployment', 'primary')}"
        diagnosis_model = patient_data.get("diagnosis_model", "Standard diagnostic protocol")
        verification_model = patient_data.get("verification_model", "Pending verification")
        
        # Format the report with medical styling
        diagnostic_info = f"""
    ==========================================
    MEDICAL ASSESSMENT REPORT
    ==========================================
    Date: {current_time}
    Report ID: {user_id[:8] if len(user_id) > 8 else user_id}-{int(time.time()) % 10000}
    Assessment Status: {current_state.replace('_', ' ').title()}
    
    ------------------------------------------
    PATIENT INFORMATION
    ------------------------------------------
    Name: {patient_name}
    Age: {patient_age}
    Gender: {patient_gender}
    Height: {patient_height}
    Weight: {patient_weight}
    
    ------------------------------------------
    PRESENTING SYMPTOMS
    ------------------------------------------
    {symptoms_text}
    
    ------------------------------------------
    ASSESSMENT NOTES
    ------------------------------------------
    {assessment_text}
    
    ------------------------------------------
    DIAGNOSTIC IMPRESSION
    ------------------------------------------
    Preliminary Diagnosis: {diagnosis_name}
    Diagnostic Confidence: {confidence_text}
    Assessment Basis: {confidence_reasoning}
    Verification Status: {'Complete - High Confidence' if verification_trigger == 'high_confidence' else 'Complete - Low Confidence' if verification_trigger == 'low_confidence' else 'Pending'}
    
    ------------------------------------------
    RECOMMENDATIONS
    ------------------------------------------
    {recommendations}
    
    ------------------------------------------
    TECHNICAL APPENDIX
    ------------------------------------------
    Analysis Protocol: {context_model}
    Diagnostic Protocol: {diagnosis_model}
    Verification Protocol: {verification_model}
    
    IMPORTANT DISCLAIMER: This is a demonstration of AI capabilities for
    medical assistant technology and is NOT actual medical advice.
    This system is for research and development purposes only.
    In a real scenario, always consult with qualified healthcare
    professionals for any medical concerns or conditions.
    ==========================================
    """
        return diagnostic_info
        
    async def process_message(self, user_id: str, message: str, include_diagnostics: bool = True) -> str:
        """
        Process a user message and return the response with diagnostic information.
        
        Args:
            user_id: The user's identifier
            message: The user's message
            include_diagnostics: Whether to include diagnostic info in the response
                
        Returns:
            Bot response text with optional diagnostic information
        """
        # Get user's history and data
        history = self.get_chat_history(user_id)
        user_data = self.get_user_data(user_id)
        
        # Ensure the patient_data structure is present
        patient_data = initialize_patient_data_if_needed(user_data)
        
        # Attempt to parse any demographic lines in the user's message
        parse_and_update_demographics(patient_data, message)
        
        # Store the original message for context in out_of_scope handling
        self.dialog_manager.store_original_message(user_id, message)
        
        # Add user message to history
        history.add_user_message(message)
        
        # Generate user context from conversation history
        user_context = await self._generate_user_context(history)
        
        # Classify intent using the LLM-based classifier
        intents = await self.intent_classifier.classify_intent(message)
        top_intent = max(intents.items(), key=lambda x: x[1])[0]
        top_score = max(intents.items(), key=lambda x: x[1])[1]
        
        logger.info(f"User message: {message}")
        logger.info(f"Current state: {self.dialog_manager.get_user_state(user_id)}")
        logger.info(f"Classified intent: {top_intent} (score: {top_score:.2f})")
        
        # If the user is informing symptoms, extract and add them
        if top_intent == "inform_symptoms" or top_intent == "symptomReporting":
            # Extract just the symptom part from the message
            # For now, we'll use a simple approach - the first symptom-like phrase
            symptom_text = message.lower()
            # Remove common prefixes that aren't part of the symptom
            prefixes_to_remove = [
                "i have ", "i've got ", "i am having ", "i'm having ",
                "i feel ", "i'm feeling ", "i am feeling ",
                "i got ", "i've been having ", "experiencing ",
                "suffering from ", "dealing with "
            ]
            for prefix in prefixes_to_remove:
                if symptom_text.startswith(prefix):
                    symptom_text = symptom_text[len(prefix):]
                    break
            
            # Remove common suffixes that aren't part of the symptom
            suffixes_to_remove = [
                " lately", " recently", " for a while", " since yesterday",
                " since last week", " for the past", " for a few days"
            ]
            for suffix in suffixes_to_remove:
                if symptom_text.endswith(suffix):
                    symptom_text = symptom_text[:-len(suffix)]
                    break
            
            # Clean up any remaining punctuation
            symptom_text = symptom_text.strip('.,!? \t\n')
            
            # Only add if we have a non-empty symptom
            if symptom_text:
                logger.info(f"Adding symptom: '{symptom_text}' to patient data")
                self.diagnostic_engine.add_symptom(patient_data, symptom_text)
                await self.diagnostic_engine.update_diagnosis_confidence(patient_data)
                logger.info(f"After adding symptom, patient data symptoms: {patient_data.get('symptoms', [])}")
                logger.info(f"Diagnosis confidence updated to: {patient_data.get('diagnosis', {}).get('confidence', 0.0)}")
                # Possibly move to verification if confidence is high enough
                if self.dialog_manager.should_verify_symptoms(user_id, patient_data):
                    logger.info("Confidence threshold reached, transitioning to verification")
        
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

# External functions for API and CLI interfaces
_bot_instances = {}

async def process_message_api(message: str, user_id: str = None, include_diagnostics: bool = True) -> str:
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
    
    print("\nInitializing bot...")
    bot = MedicalAssistantBot()
    user_id = "interactive_user"
    
    include_diagnostics = True
    
    print("\n----- Starting Interactive Medical Assistant Conversation -----")
    print("Type your messages and press Enter.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Type 'debug on/off' to toggle diagnostic information.")
    print("Type 'help' for more commands.\n")
    
    print("Bot: Hello! I'm your medical assistant. How can I help you today?")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting conversation due to keyboard interrupt.")
            break
            
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nBot: Thank you for talking with me. Take care!")
            break
        elif user_input.lower() == "help":
            print("\n----- Commands -----")
            print("exit, quit, bye - End the conversation")
            print("debug on/off - Toggle diagnostic information")
            print("clear - Clear the conversation history")
            print("status - Check bot status")
            print("------------------")
            continue
        elif user_input.lower() in ["debug on", "debug true"]:
            include_diagnostics = True
            print("\nDiagnostic information enabled.")
            continue
        elif user_input.lower() in ["debug off", "debug false"]:
            include_diagnostics = False
            print("\nDiagnostic information disabled.")
            continue
        elif user_input.lower() == "clear":
            if user_id in bot.chat_histories:
                bot.chat_histories[user_id] = ChatHistory()
                bot.user_data[user_id] = {}
                print("\nConversation history cleared.")
            continue
        elif user_input.lower() == "status":
            print("\n----- Bot Status -----")
            print(f"Dialog State: {bot.dialog_manager.get_user_state(user_id)}")
            
            if user_id in bot.user_data and "patient_data" in bot.user_data[user_id]:
                pd = bot.user_data[user_id]["patient_data"]
                print("Patient Data Model:")
                print(pd)
            print("---------------------")
            continue
        
        if not user_input:
            continue
            
        try:
            print("Bot: Thinking...", end="\r")
            logger.info(f"Processing user input: {user_input}")
            
            response = await bot.process_message(user_id, user_input, include_diagnostics=include_diagnostics)
            print(" " * 50, end="\r")
            
            print(f"\nBot: {response}")
                
        except Exception as e:
            print(" " * 50, end="\r")
            error_msg = str(e)
            print(f"\n\u001b[31mError processing message: {error_msg}\u001b[0m")
            logger.error(f"Error processing message: {error_msg}")
            logger.error(traceback.format_exc())
            print("\nBot: I'm sorry, I encountered an error. Please check the logs for details or try again.")

if __name__ == "__main__":
    asyncio.run(interactive_conversation())
