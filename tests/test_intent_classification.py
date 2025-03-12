#!/usr/bin/env python3
"""
Test script to verify that the medical assistant bot correctly classifies user intents,
especially for symptom clarification responses.
"""
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_bot import MedicalAssistantBot
from intent_classifier import MedicalIntentClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_intent_classification():
    """
    Test that the intent classifier correctly classifies various user responses,
    especially symptom clarification responses.
    """
    # Initialize the intent classifier
    intent_classifier = MedicalIntentClassifier()
    user_id = "test_intent_user"
    
    # Test responses that should be classified as symptom clarifications
    symptom_clarification_responses = [
        "yes",
        "no",
        "sometimes",
        "rarely",
        "often",
        "a little bit",
        "not really",
        "no, nothing like this",
        "yes, that's correct",
        "it's very painful",
        "only when I move"
    ]
    
    # Test responses that should be classified as out of scope
    out_of_scope_responses = [
        "What's the weather like today?",
        "Can you tell me about the latest movies?",
        "Who won the football game last night?",
        "Tell me a joke",
        "What's your favorite color?"
    ]
    
    # First, add a previous question to the context
    intent_classifier.previous_questions[user_id] = "Do you experience any pain when you move your head?"
    
    # Test symptom clarification responses
    logger.info("Testing symptom clarification responses...")
    for response in symptom_clarification_responses:
        intents = await intent_classifier.classify_intent(response, user_id)
        top_intent = max(intents.items(), key=lambda x: x[1])[0]
        top_score = max(intents.items(), key=lambda x: x[1])[1]
        
        logger.info(f"Response: '{response}' -> Intent: {top_intent} (score: {top_score:.2f})")
        
        # Check if the intent is correctly classified
        if top_intent == "symptomClarification" or top_intent == "symptomReporting":
            logger.info("✅ Correctly classified as symptom-related")
        else:
            logger.error(f"❌ Incorrectly classified as {top_intent}")
    
    # Test out of scope responses
    logger.info("\nTesting out-of-scope responses...")
    for response in out_of_scope_responses:
        intents = await intent_classifier.classify_intent(response, user_id)
        top_intent = max(intents.items(), key=lambda x: x[1])[0]
        top_score = max(intents.items(), key=lambda x: x[1])[1]
        
        logger.info(f"Response: '{response}' -> Intent: {top_intent} (score: {top_score:.2f})")
        
        # Check if the intent is correctly classified
        if top_intent == "out_of_scope":
            logger.info("✅ Correctly classified as out-of-scope")
        else:
            logger.error(f"❌ Incorrectly classified as {top_intent}")

async def test_full_conversation():
    """
    Test a full conversation flow to ensure symptom clarification responses
    are properly handled and questions are marked as answered.
    """
    # Initialize the bot
    bot = MedicalAssistantBot()
    user_id = "test_conversation_user"
    
    # Start the conversation
    logger.info("\nStarting test conversation...")
    response = await bot.process_message(user_id, "Hello", include_diagnostics=False)
    logger.info(f"Bot: {response}")
    
    # Report an initial symptom
    response = await bot.process_message(user_id, "I have a headache", include_diagnostics=False)
    logger.info(f"User: I have a headache")
    logger.info(f"Bot: {response}")
    
    # Respond to follow-up questions with simple answers
    conversation = [
        "no",  # First response
        "yes",  # Second response
        "sometimes",  # Third response
        "not really",  # Fourth response
        "a little bit"  # Fifth response
    ]
    
    # Continue the conversation
    for i, user_message in enumerate(conversation):
        # Get the current state and patient data before the message
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        current_state = bot.dialog_manager.get_user_state(user_id)
        asked_questions = patient_data.get("asked_questions", [])
        answered_questions = [q for q in asked_questions if isinstance(q, dict) and q.get("is_answered", False)]
        
        logger.info(f"\nTurn {i+1}:")
        logger.info(f"Current state: {current_state}")
        logger.info(f"Total questions: {len(asked_questions)}")
        logger.info(f"Answered questions: {len(answered_questions)}")
        
        # Send the user message
        logger.info(f"User: {user_message}")
        response = await bot.process_message(user_id, user_message, include_diagnostics=False)
        logger.info(f"Bot: {response}")
        
        # Check the state after the message
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        new_state = bot.dialog_manager.get_user_state(user_id)
        asked_questions = patient_data.get("asked_questions", [])
        answered_questions = [q for q in asked_questions if isinstance(q, dict) and q.get("is_answered", False)]
        symptom_related_questions = [q for q in asked_questions if isinstance(q, dict) and q.get("is_symptom_related", False)]
        
        logger.info(f"New state: {new_state}")
        logger.info(f"Total questions: {len(asked_questions)}")
        logger.info(f"Answered questions: {len(answered_questions)}")
        logger.info(f"Symptom-related questions: {len(symptom_related_questions)}")
        logger.info(f"Symptoms: {patient_data.get('symptoms', [])}")
        logger.info(f"Diagnosis confidence: {patient_data.get('diagnosis', {}).get('confidence', 0.0)}")
        
        # If we've transitioned to verification, we're done
        if new_state == "verification":
            logger.info("✅ Successfully transitioned to verification state!")
            
            # Check verification info
            verification_info = patient_data.get("verification_info", {})
            trigger_reason = verification_info.get("trigger_reason", "")
            logger.info(f"Verification triggered by: {trigger_reason}")
            break
    
    # Final check
    final_state = bot.dialog_manager.get_user_state(user_id)
    if final_state == "verification":
        logger.info("\n✅ TEST PASSED: Bot correctly transitioned to verification state")
    else:
        logger.error(f"\n❌ TEST FAILED: Bot did not transition to verification state. Final state: {final_state}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the tests
    logger.info("Starting intent classification tests")
    asyncio.run(test_intent_classification())
    asyncio.run(test_full_conversation())
    logger.info("Tests completed")
