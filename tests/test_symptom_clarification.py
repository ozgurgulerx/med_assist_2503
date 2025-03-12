#!/usr/bin/env python3
"""
Test script to verify that the medical assistant bot correctly handles symptom clarification responses.
"""
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_bot import MedicalAssistantBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_symptom_clarification():
    """
    Test that the bot correctly handles symptom clarification responses.
    """
    # Initialize the bot
    bot = MedicalAssistantBot()
    user_id = "test_symptom_clarification_user"
    
    # Start the conversation
    logger.info("Starting conversation with greeting")
    response = await bot.process_message(user_id, "Hello", include_diagnostics=False)
    logger.info(f"Bot: {response}")
    
    # Report an initial symptom
    logger.info("Reporting initial symptom")
    response = await bot.process_message(user_id, "I have a headache", include_diagnostics=False)
    logger.info(f"Bot: {response}")
    
    # Get the current state and patient data
    user_data = bot.get_user_data(user_id)
    patient_data = user_data.get("patient_data", {})
    current_state = bot.dialog_manager.get_user_state(user_id)
    logger.info(f"Current state: {current_state}")
    logger.info(f"Symptoms: {patient_data.get('symptoms', [])}")
    
    # Respond to follow-up questions with simple answers
    test_responses = [
        "no, nothing like this",
        "yes",
        "sometimes",
        "not really",
        "a little bit"
    ]
    
    # Keep track of the number of questions asked and answered
    max_iterations = 10  # Safety limit to prevent actual infinite loops during testing
    
    # Continue the conversation until we transition to verification or reach max iterations
    for i in range(max_iterations):
        # Get the current state
        current_state = bot.dialog_manager.get_user_state(user_id)
        logger.info(f"Current state (iteration {i+1}): {current_state}")
        
        # If we've transitioned to verification, we're done
        if current_state == "verification":
            logger.info("Successfully transitioned to verification state!")
            break
        
        # Send a simple response
        test_response = test_responses[i % len(test_responses)]
        logger.info(f"Sending response: {test_response}")
        response = await bot.process_message(user_id, test_response, include_diagnostics=False)
        logger.info(f"Bot: {response}")
        
        # Check the questions
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        asked_questions = patient_data.get("asked_questions", [])
        
        # Count answered and symptom-related questions
        answered_questions = [q for q in asked_questions if isinstance(q, dict) and q.get("is_answered", False)]
        symptom_related_questions = [q for q in asked_questions if isinstance(q, dict) and q.get("is_symptom_related", False)]
        
        logger.info(f"Total questions: {len(asked_questions)}")
        logger.info(f"Answered questions: {len(answered_questions)}")
        logger.info(f"Symptom-related questions: {len(symptom_related_questions)}")
        
        # Get the confidence
        confidence = patient_data.get("diagnosis", {}).get("confidence", 0.0)
        logger.info(f"Current diagnosis confidence: {confidence}")
    
    # Final check
    final_state = bot.dialog_manager.get_user_state(user_id)
    if final_state == "verification":
        logger.info("TEST PASSED: Bot correctly transitioned to verification state")
        
        # Check verification info
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        verification_info = patient_data.get("verification_info", {})
        trigger_reason = verification_info.get("trigger_reason", "")
        
        logger.info(f"Verification triggered by: {trigger_reason}")
    else:
        logger.error(f"TEST FAILED: Bot did not transition to verification state. Final state: {final_state}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the test
    logger.info("Starting symptom clarification test")
    asyncio.run(test_symptom_clarification())
    logger.info("Test completed")
