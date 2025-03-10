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
                    "inform_symptoms": "collecting_symptoms",
                    "ask_medical_info": "providing_info",
                    "out_of_scope": "out_of_scope_handler"
                }
            },
            "collecting_symptoms": {
                "next_actions": ["action_ask_followup_question"],
                "transitions": {
                    # Repeated inform_symptoms => stay collecting
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
                    # e.g. confirm => generating_diagnosis
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
                    # All intents lead back to greeting or other relevant states
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

        # Current dialog state for each user
        self.user_states: Dict[str, str] = {}

        # Store the original user message for context
        self.original_messages: Dict[str, str] = {}

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

        # Lookup transitions
        state_info = self.dialog_states.get(current_state, {})
        transitions = state_info.get("transitions", {})

        # Handle symptom reporting intent specifically
        if intent == "symptomReporting":
            next_state = "collecting_symptoms"  # Force transition for symptom reporting
        elif intent in transitions:
            next_state = transitions[intent]
        else:
            next_state = current_state  # Fallback to current state if intent not recognized

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
        question_count = len(patient_data.get("asked_questions", []))

        # Need at least 1 symptom and either:
        # - >= 0.85 confidence, or
        # - >= 3 rounds of follow-up
        should_verify = (len(symptoms) >= 1 and (confidence >= 0.85 or question_count >= 3))

        if should_verify:
            logger.info(
                f"Transitioning to verification for user {user_id} "
                f"(confidence: {confidence:.2f}, questions: {question_count})"
            )
            self.set_user_state(user_id, "verification")

        return should_verify
