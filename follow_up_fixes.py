# Changes to implement in core_bot.py

# 1. Modify the process_message method to track unanswered symptom questions and steer conversation back

'''
Add this at the beginning of process_message method (around line 600):
'''

# Store the last unanswered symptom question for reference
last_unanswered_symptom_question = None
asked_questions = patient_data.get("asked_questions", [])
for q in reversed(asked_questions):
    if isinstance(q, dict) and q.get("is_symptom_related", False) and not q.get("is_answered", False):
        last_unanswered_symptom_question = q
        break

'''
Add this after the out_of_scope handling (around line 650):
'''

# For other intents that aren't symptom clarification, check if we need to steer back to an unanswered symptom question
elif top_intent not in ["symptomClarification", "greeting", "farewell"] and last_unanswered_symptom_question is not None:
    # We have an unanswered symptom question and the user asked something unrelated
    # We'll steer the conversation back to the symptom question
    steer_back = True
    current_state = self.dialog_manager.get_user_state(user_id)
    
    # Only steer back if we're in a state where we should be collecting symptoms
    if current_state in ["collecting_symptoms", "asking_followup"]:
        logger.info(f"Steering conversation back to unanswered symptom question: {last_unanswered_symptom_question['question']}")
        # We'll handle this in the response generation
    else:
        steer_back = False
else:
    steer_back = False

'''
Modify the response generation section (around line 690) to handle steering back to symptom questions:
'''

# Generate response based on the current state
response_text = ""
technical_appendix = ""
model_info = {}

# If we need to steer back to an unanswered symptom question
if steer_back and last_unanswered_symptom_question is not None:
    response_text = f"I understand, but first, could you please answer this question about your symptoms: {last_unanswered_symptom_question['question']}"
    model_info = {"model": "gpt-4o", "deployment": "mini"}
else:
    # Normal response generation flow...
    # (keep existing response generation code here)

# 2. Modify the diagnostic_engine.py to ensure follow-up count isn't increased incorrectly

'''
In diagnostic_engine.py, modify the generate_followup_question method to ensure we're not adding duplicate questions:
'''

# Before adding a new question, check if there's already an unanswered symptom question
unanswered_symptom_question = None
for q in asked_list:
    if isinstance(q, dict) and q.get("is_symptom_related", True) and not q.get("is_answered", False):
        unanswered_symptom_question = q
        break

# Only add a new question if there are no unanswered symptom questions
if unanswered_symptom_question is None:
    # Add the new question (existing code)
    asked_dict = {
        "question": question_text,
        "answer": "",
        "timestamp_asked": current_utc_timestamp(),
        "timestamp_answered": "",
        "is_symptom_related": True,
        "is_answered": False
    }
    asked_list.append(asked_dict)
else:
    # Use the existing unanswered question
    question_text = unanswered_symptom_question["question"]
    logger.info(f"Reusing existing unanswered symptom question: {question_text}")

# 3. Modify the dialog_manager.py to ensure state changes don't reset the count

'''
In dialog_manager.py, modify the transition_state method to preserve patient data across state changes:
'''

def transition_state(self, user_id: str, new_state: str, patient_data: Dict[str, Any]) -> None:
    """Transition the user to a new state while preserving patient data."""
    old_state = self.get_user_state(user_id)
    logger.info(f"Transitioning user {user_id} from state {old_state} to {new_state}")
    
    # Store the current state in the user_states dictionary
    self.user_states[user_id] = new_state
    
    # Ensure we don't reset patient data during state transitions
    # The patient_data should be passed by reference and maintained by the caller
    
    # Log the transition for debugging
    logger.info(f"State transition complete. User {user_id} now in state {new_state}")
    logger.info(f"Patient data preserved with {len(patient_data.get('asked_questions', []))} questions")
