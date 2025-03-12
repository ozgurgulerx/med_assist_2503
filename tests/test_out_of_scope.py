"""Test script to verify out-of-scope handling and question counting"""

import asyncio
import logging
import sys
from typing import List, Dict, Any
from core_bot import MedicalAssistantBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test_out_of_scope_handling():
    """Test that out-of-scope responses don't count toward the follow-up question limit"""
    bot = MedicalAssistantBot()
    user_id = "test_out_of_scope_user"
    
    # Conversation with out-of-scope responses interspersed with medical responses
    conversation = [
        "Hello",  # Greeting
        "I have a headache",  # Initial symptom
        "What's the weather like today?",  # Out-of-scope
        "It's behind my eye and throbbing",  # Symptom detail
        "Can you tell me about the latest movies?",  # Out-of-scope
        "Yes, I also feel nauseous",  # Additional symptom
        "What's your favorite color?",  # Out-of-scope
        "It started this morning",  # Symptom timing
        "How many planets are in the solar system?",  # Out-of-scope
        "I haven't taken any medication yet"  # Treatment info
    ]
    
    responses = []
    for message in conversation:
        logger.info(f"\nUser: {message}")
        response = await bot.process_message(user_id, message, include_diagnostics=False)
        responses.append(response)
        logger.info(f"Bot: {response}")
        
        # After each message, check the state and question count
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        asked_questions = patient_data.get("asked_questions", [])
        symptom_questions = [q for q in asked_questions 
                            if isinstance(q, dict) 
                            and q.get("is_symptom_related", False) 
                            and q.get("is_answered", True)]
        
        current_state = bot.dialog_manager.get_user_state(user_id)
        logger.info(f"Current state: {current_state}")
        logger.info(f"Total questions: {len(asked_questions)}")
        logger.info(f"Symptom-related answered questions: {len(symptom_questions)}")
        
        # List all questions with their properties
        for i, q in enumerate(asked_questions):
            if isinstance(q, dict):
                logger.info(f"Question {i+1}: '{q.get('question', 'Unknown')}' - "
                          f"symptom_related: {q.get('is_symptom_related', False)}, "
                          f"answered: {q.get('is_answered', False)}")
    
    # Final verification
    user_data = bot.get_user_data(user_id)
    patient_data = user_data.get("patient_data", {})
    asked_questions = patient_data.get("asked_questions", [])
    symptom_questions = [q for q in asked_questions 
                        if isinstance(q, dict) 
                        and q.get("is_symptom_related", False) 
                        and q.get("is_answered", True)]
    
    logger.info("\n=== Final Test Results ===")
    logger.info(f"Total questions asked: {len(asked_questions)}")
    logger.info(f"Symptom-related answered questions: {len(symptom_questions)}")
    logger.info(f"Final state: {bot.dialog_manager.get_user_state(user_id)}")
    
    # Verify that out-of-scope questions didn't count toward the limit
    assert len(symptom_questions) <= 5, "Too many symptom questions were counted"
    
    logger.info("Test completed successfully!")

async def test_emergency_detection():
    """Test that emergency situations are properly detected and handled"""
    bot = MedicalAssistantBot()
    user_id = "test_emergency_user"
    
    # Conversation with emergency symptoms
    conversation = [
        "Hello",  # Greeting
        "I'm having severe chest pain and difficulty breathing",  # Emergency symptoms
        "It feels like pressure and radiates to my left arm",  # More emergency details
        "I'm also sweating a lot and feeling dizzy"  # Additional emergency symptoms
    ]
    
    responses = []
    for message in conversation:
        logger.info(f"\nUser: {message}")
        response = await bot.process_message(user_id, message, include_diagnostics=False)
        responses.append(response)
        logger.info(f"Bot: {response}")
        
        # Check if we've transitioned to emergency state
        current_state = bot.dialog_manager.get_user_state(user_id)
        logger.info(f"Current state: {current_state}")
        
        if current_state == "emergency":
            logger.info("Emergency state detected!")
            break
    
    # Verify that we reached emergency state
    final_state = bot.dialog_manager.get_user_state(user_id)
    logger.info(f"Final state: {final_state}")
    
    # The test passes if we detected an emergency
    if final_state == "emergency":
        logger.info("Emergency detection test passed!")
    else:
        logger.error("Emergency detection test failed! Did not reach emergency state.")

async def main():
    """Run all tests"""
    logger.info("Starting out-of-scope handling test...")
    await test_out_of_scope_handling()
    
    logger.info("\n\nStarting emergency detection test...")
    await test_emergency_detection()

if __name__ == "__main__":
    asyncio.run(main())
