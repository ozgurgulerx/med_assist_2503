"""
Dialog manager for the medical assistant bot
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DialogManager:
    """Manages dialog states and transitions for the medical assistant"""
    
    def __init__(self):
        """Initialize the dialog manager"""
        # Define dialog states
        self.dialog_states = {
            "greeting": {
                "next_actions": ["utter_greet", "utter_how_can_i_help"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "out_of_scope": "out_of_scope_handler"
                }
            },
            "collecting_symptoms": {
                "next_actions": ["action_ask_followup_question"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "deny": "generating_diagnosis",
                    "ask_medical_info": "providing_info",
                    "goodbye": "farewell",
                    "out_of_scope": "out_of_scope_handler"
                }
            },
            "providing_info": {
                "next_actions": ["action_provide_medical_info", "utter_anything_else"],
                "transitions": {
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "deny": "farewell",
                    "goodbye": "farewell",
                    "out_of_scope": "out_of_scope_handler"
                }
            },
            "verification": {
                "next_actions": ["action_verify_symptoms"],
                "transitions": {
                    "confirm": "generating_diagnosis",
                    "deny": "collecting_symptoms",
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info"
                }
            },
            "generating_diagnosis": {
                "next_actions": ["action_provide_diagnosis", "utter_suggest_mitigations"],
                "transitions": {
                    "ask_medical_info": "providing_info",
                    "confirm": "farewell",
                    "goodbye": "farewell",
                    "out_of_scope": "out_of_scope_handler"
                }
            },
            "farewell": {
                "next_actions": ["utter_goodbye"],
                "transitions": {
                    "out_of_scope": "out_of_scope_handler"
                }
            },
            "out_of_scope_handler": {
                "next_actions": ["action_handle_out_of_scope", "utter_redirect_to_medical"],
                "transitions": {
                    # All intents will transition back to greeting
                    "greet": "greeting",
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "confirm": "greeting",
                    "deny": "greeting",
                    "goodbye": "farewell",
                    "out_of_scope": "greeting"
                }
            }
        }
        
        # Current state for each user
        self.user_states: Dict[str, str] = {}
        
        # Store the original user message for context
        self.original_messages: Dict[str, str] = {}
        
    def get_user_state(self, user_id: str) -> str:
        """Get the current dialog state for a user"""
        if user_id not in self.user_states:
            self.user_states[user_id] = "greeting"
        return self.user_states[user_id]
    
    def set_user_state(self, user_id: str, state: str) -> None:
        """Set the dialog state for a user"""
        self.user_states[user_id] = state
        logger.info(f"Set user {user_id} state to: {state}")
    
    def store_original_message(self, user_id: str, message: str) -> None:
        """Store the original message for context in out_of_scope handling"""
        self.original_messages[user_id] = message
    
    def get_original_message(self, user_id: str) -> str:
        """Get the original message for context"""
        return self.original_messages.get(user_id, "")
    
    def get_next_state(self, user_id: str, intent: str) -> str:
        """
        Determine the next state based on current state and intent
        
        Args:
            user_id: The user's identifier
            intent: The classified intent
        
        Returns:
            The next state to transition to
        """
        current_state = self.get_user_state(user_id)
        state_info = self.dialog_states.get(current_state, {})
        transitions = state_info.get("transitions", {})
        
        # Log the intent and current state for debugging
        logger.info(f"Determining next state from {current_state} with intent: {intent}")
        
        # Special case: if we're collecting symptoms and get another inform_symptoms intent,
        # we should stay in the collecting_symptoms state
        if current_state == "collecting_symptoms" and intent == "inform_symptoms":
            logger.info(f"User {user_id} provided more symptoms, staying in collecting_symptoms state")
            return "collecting_symptoms"
        
        # Normal transition based on intent
        next_state = transitions.get(intent, current_state)
        
        # If we're in verification and the user confirms, move to diagnosis
        if current_state == "verification" and intent == "confirm":
            next_state = "generating_diagnosis"
        
        # If we're in verification and the user denies or provides more symptoms,
        # go back to collecting symptoms
        if current_state == "verification" and (intent == "deny" or intent == "inform_symptoms"):
            next_state = "collecting_symptoms"
        
        logger.info(f"Transitioning user {user_id} from {current_state} to {next_state} (intent: {intent})")
        return next_state
    
    def get_next_actions(self, state: str) -> List[str]:
        """
        Get the next actions for a given state
        
        Args:
            state: The dialog state
        
        Returns:
            List of action names to execute
        """
        return self.dialog_states.get(state, {}).get("next_actions", [])
    
    def should_verify_symptoms(self, user_id: str, patient_data: Dict[str, Any]) -> bool:
        """
        Determine if we should transition to verification state
        based on symptom confidence
        
        Args:
            user_id: The user's identifier
            patient_data: Patient data including symptoms and confidence
            
        Returns:
            Boolean indicating if we should verify
        """
        current_state = self.get_user_state(user_id)
        
        # Only consider verification from collecting_symptoms state
        if current_state != "collecting_symptoms":
            return False
            
        # Check if we have enough symptoms and confidence
        symptoms = patient_data.get("symptoms", [])
        confidence = patient_data.get("diagnosis_confidence", 0.0)
        
        # Count how many rounds of follow-up questions we've had
        question_count = len(patient_data.get("asked_questions", []))
        
        # Need at least 1 symptom and either:
        # - 60% confidence, or
        # - At least 3 rounds of follow-up questions
        should_verify = (len(symptoms) >= 1 and (confidence >= 0.6 or question_count >= 3))
        
        if should_verify:
            logger.info(f"Transitioning to verification for user {user_id} (confidence: {confidence:.2f}, questions: {question_count})")
            self.set_user_state(user_id, "verification")
            
        return should_verify