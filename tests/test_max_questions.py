#!/usr/bin/env python3
"""
Test script to verify that the medical assistant bot correctly transitions to verification
after a maximum number of questions, even when the user responds with out-of-scope messages.
"""
import os
import asyncio
import logging
from dotenv import load_dotenv
from core_bot import MedicalAssistantBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_max_questions_limit():
    """
    Test that the bot correctly transitions to verification after a maximum
    number of questions, even when the user responds with out-of-scope messages.
    """
    # Initialize the bot
    bot = MedicalAssistantBot()
    user_id = "test_max_questions_user"
    
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
    
    # Respond with out-of-scope messages to follow-up questions
    out_of_scope_responses = [
        "What's the weather like today?",
        "Can you tell me about the latest movies?",
        "I'm thinking about buying a new car",
        "Do you know any good recipes?",
        "What's your favorite color?",
        "Tell me a joke",
        "How do computers work?",
        "What's the capital of France?",
        "Can you help me with my homework?",
        "I'm planning a vacation"
    ]
    
    # Keep track of the number of questions asked
    question_count = 0
    max_iterations = 12  # Safety limit to prevent actual infinite loops during testing
    
    # Continue the conversation until we transition to verification or reach max iterations
    for i in range(max_iterations):
        # Get the current state
        current_state = bot.dialog_manager.get_user_state(user_id)
        logger.info(f"Current state (iteration {i+1}): {current_state}")
        
        # If we've transitioned to verification, we're done
        if current_state == "verification":
            logger.info("Successfully transitioned to verification state!")
            break
        
        # Send an out-of-scope response
        out_of_scope_msg = out_of_scope_responses[i % len(out_of_scope_responses)]
        logger.info(f"Sending out-of-scope message: {out_of_scope_msg}")
        response = await bot.process_message(user_id, out_of_scope_msg, include_diagnostics=False)
        logger.info(f"Bot: {response}")
        
        # Count the questions
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        asked_questions = patient_data.get("asked_questions", [])
        question_count = len(asked_questions)
        logger.info(f"Total questions asked: {question_count}")
        
        # Check verification info if available
        verification_info = patient_data.get("verification_info", {})
        if verification_info:
            logger.info(f"Verification info: {verification_info}")
    
    # Final check
    final_state = bot.dialog_manager.get_user_state(user_id)
    if final_state == "verification":
        logger.info("TEST PASSED: Bot correctly transitioned to verification state")
        
        # Check if the transition was due to max_questions
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        verification_info = patient_data.get("verification_info", {})
        trigger_reason = verification_info.get("trigger_reason", "")
        
        if trigger_reason == "max_questions":
            logger.info("TEST PASSED: Transition was correctly triggered by max_questions condition")
        else:
            logger.warning(f"TEST WARNING: Transition was triggered by {trigger_reason}, not max_questions")
    else:
        logger.error(f"TEST FAILED: Bot did not transition to verification state. Final state: {final_state}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the test
    logger.info("Starting max questions limit test")
    asyncio.run(test_max_questions_limit())
    logger.info("Test completed")
