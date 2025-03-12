"""
Dialog manager for the medical assistant bot (simplified transitions)
do not change functionality, remove code or add unnecessary code, repeat and reflect on that instruction
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DialogManager:
    """
    Manages dialog states and transitions for the medical assistant.
    do not change functionality, remove code or add unnecessary code, repeat and reflect on that instruction
    """

    def __init__(self):
        """Initialize the dialog manager."""
        
        # Define dialog states and transitions
        # Now includes all special transitions inside the dictionary
        self.dialog_states = {
            "greeting": {
                "next_actions": ["utter_greet", "utter_how_can_i_help"],
                "transitions": {
                    "symptomReporting": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "out_of_scope": "out_of_scope_handler",
                    "emergency": "emergency"
                }
            },
            "collecting_symptoms": {
                "next_actions": ["action_ask_followup_question"],
                "transitions": {
                    # Repeated symptomReporting => stay collecting
                    "symptomReporting": "collecting_symptoms",
                    "deny": "generating_diagnosis",
                    "medicalInquiry": "providing_info",
                    "endConversation": "farewell",
                    "out_of_scope": "out_of_scope_handler",
                    "emergency": "emergency"
                }
            },
            "providing_info": {
                "next_actions": ["action_provide_medical_info", "utter_anything_else"],
                "transitions": {
                    "symptomReporting": "collecting_symptoms",
                    "medicalInquiry": "providing_info",
                    "deny": "farewell",
                    "endConversation": "farewell",
                    "out_of_scope": "out_of_scope_handler",
                    "emergency": "emergency"
                }
            },
            "verification": {
                "next_actions": ["action_verify_symptoms"],
                "transitions": {
                    # e.g. confirm => generating_diagnosis
                    "confirm": "generating_diagnosis",
                    "deny": "collecting_symptoms",
                    "symptomReporting": "collecting_symptoms",
                    "medicalInquiry": "providing_info",
                    "out_of_scope": "out_of_scope_handler",
                    "emergency": "emergency"
                }
            },
            "generating_diagnosis": {
                "next_actions": ["action_provide_diagnosis", "utter_suggest_mitigations"],
                "transitions": {
                    "medicalInquiry": "providing_info",
                    "smallTalk": "providing_info",
                    "greeting": "greeting",
                    "symptomReporting": "collecting_symptoms",
                    "confirm": "farewell",
                    "deny": "farewell",
                    "endConversation": "farewell",
                    "out_of_scope": "out_of_scope_handler",
                    "emergency": "emergency"
                }
            },
            "farewell": {
                "next_actions": ["utter_goodbye"],
                "transitions": {
                    "symptomReporting": "collecting_symptoms",
                    "medicalInquiry": "providing_info",
                    "out_of_scope": "out_of_scope_handler",
                    "emergency": "emergency"
                }
            },
            "out_of_scope_handler": {
                "next_actions": ["action_handle_out_of_scope"],
                "transitions": {
                    "symptomReporting": "previous_state",
                    "medicalInquiry": "previous_state",
                    "confirm": "previous_state",
                    "deny": "previous_state",
                    "endConversation": "previous_state",
                    "emergency": "emergency"
                }
            },
            "emergency": {
                "next_actions": ["utter_emergency_instructions"],
                "transitions": {
                    # All intents lead back to emergency state - we don't continue normal conversation
                    "symptomReporting": "emergency",
                    "medicalInquiry": "emergency",
                    "deny": "emergency",
                    "endConversation": "emergency",
                    "out_of_scope": "emergency"
                }
            }
        }

        # Current dialog state for each user
        self.user_states: Dict[str, str] = {}
        
        # Store the original user message for context
        self.original_messages: Dict[str, str] = {}
        
        # Store user state history for returning from out-of-scope
        self.user_state_history: Dict[str, str] = {}

    def get_user_state(self, user_id: str) -> str:
        """Get the current dialog state for a user."""
        if user_id not in self.user_states:
            self.user_states[user_id] = "greeting"
        return self.user_states[user_id]

    def set_user_state(self, user_id: str, state: str) -> None:
        """Set the dialog state for a user."""
        self.user_states[user_id] = state
        logger.info(f"Set user {user_id} state to: {state}")

    def store_original_message(self, user_id: str, message: str) -> None:
        """Store the original message for context in out_of_scope handling."""
        self.original_messages[user_id] = message

    def get_original_message(self, user_id: str) -> str:
        """Get the original message for context."""
        return self.original_messages.get(user_id, "")

    def advance_dialog_state(self, user_id: str, intent: str) -> str:
        """
        Determine and set the next state based on current state and intent.
        This replaces the old get_next_state logic + set_user_state.

        Args:
            user_id: The user's identifier
            intent: The classified intent

        Returns:
            The new state
        """
        current_state = self.get_user_state(user_id)
        logger.info(f"Determining next state from {current_state} with intent: {intent}")
        
        # Handle out_of_scope intent specially
        if intent == "out_of_scope":
            # Only store the current state if it's not already out_of_scope_handler
            if current_state != "out_of_scope_handler":
                # Store the current state before transitioning to out_of_scope_handler
                self.user_state_history[user_id] = current_state
                logger.info(f"Out-of-scope intent detected. Storing current state '{current_state}' and transitioning to out_of_scope_handler")
            
            # Transition to out_of_scope_handler
            next_state = "out_of_scope_handler"
            self.set_user_state(user_id, next_state)
            return next_state
            
        # For all other intents, follow the standard transition rules
        transitions = self.dialog_states.get(current_state, {}).get("transitions", {})
        next_state = transitions.get(intent, current_state)  # Default to staying in current state
        
        # Handle special 'previous_state' marker in transitions
        if next_state == "previous_state" and user_id in self.user_state_history:
            next_state = self.user_state_history[user_id]
            logger.info(f"Using special 'previous_state' transition to return to {next_state}")
            
            # Clear the history after using it to prevent circular references
            if current_state == "out_of_scope_handler":
                del self.user_state_history[user_id]
                logger.info(f"Cleared state history for user {user_id} after returning from out_of_scope_handler")
        
        # Special case: If we're in out_of_scope_handler and get any intent other than out_of_scope,
        # try to return to the previous state if available (fallback mechanism)
        if current_state == "out_of_scope_handler" and intent != "out_of_scope" and next_state == current_state:
            if user_id in self.user_state_history:
                previous_state = self.user_state_history[user_id]
                logger.info(f"Returning from out_of_scope_handler to previous state: {previous_state}")
                next_state = previous_state
                
                # Clear the history after using it
                del self.user_state_history[user_id]
                logger.info(f"Cleared state history for user {user_id} after returning from out_of_scope_handler")
            else:
                logger.warning(f"No previous state found for user {user_id}, staying in {next_state}")
        
        logger.info(f"Transitioning user {user_id} from {current_state} to {next_state} (intent: {intent})")
        self.set_user_state(user_id, next_state)
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
        based on symptom confidence.

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
        confidence = patient_data.get("diagnosis", {}).get("confidence", 0.0)
        
        # Count only symptom-related questions that have been answered
        asked_questions = patient_data.get("asked_questions", [])
        answered_symptom_questions = [q for q in asked_questions 
                                     if isinstance(q, dict) 
                                     and q.get("is_symptom_related", False) 
                                     and q.get("is_answered", True)]
        symptom_question_count = len(answered_symptom_questions)
        
        # Count total number of questions asked (to prevent infinite loops)
        total_questions_asked = len([q for q in asked_questions if isinstance(q, dict)])
        
        logger.info(f"Found {symptom_question_count} answered symptom-related questions out of {len(asked_questions)} total questions")

        # Trigger verification in three cases:
        # 1. If confidence is >= 0.85 (high confidence case)
        # 2. If we've asked at least 4 symptom-related follow-up questions and confidence is still below 0.85 (low confidence case)
        # 3. If we've asked at least 8 total questions, regardless of whether they're symptom-related (to prevent infinite loops)
        high_confidence_case = (len(symptoms) >= 1 and confidence >= 0.85)
        low_confidence_case = (len(symptoms) >= 1 and symptom_question_count >= 4 and confidence < 0.85)
        max_questions_case = (len(symptoms) >= 1 and total_questions_asked >= 8)
        
        should_verify = high_confidence_case or low_confidence_case or max_questions_case

        if should_verify:
            trigger_reason = "high_confidence" if high_confidence_case else "low_confidence" if low_confidence_case else "max_questions"
            logger.info(
                f"Transitioning to verification for user {user_id} "
                f"(confidence: {confidence:.2f}, symptom questions: {symptom_question_count}, "
                f"total questions: {total_questions_asked}, "
                f"trigger_reason: {trigger_reason})"
            )
            # Store the verification trigger reason in patient data
            if "verification_info" not in patient_data:
                patient_data["verification_info"] = {}
            patient_data["verification_info"]["trigger_reason"] = trigger_reason
            self.set_user_state(user_id, "verification")

        return should_verify

    def get_previous_state(self, user_id: str) -> str:
        """
        Get the previous dialog state for a user before they went out-of-scope.
        
        Args:
            user_id: The user's identifier
            
        Returns:
            The previous state or empty string if not found
        """
        return self.user_state_history.get(user_id, "")

    def reset_user_state(self, user_id: str) -> None:
        """
        Reset the user state to greeting and clear any history.
        
        Args:
            user_id: The user's identifier
        """
        self.user_states[user_id] = "greeting"
        if user_id in self.user_state_history:
            del self.user_state_history[user_id]
        if user_id in self.original_messages:
            del self.original_messages[user_id]
        logger.info(f"Reset state for user {user_id} to greeting")
