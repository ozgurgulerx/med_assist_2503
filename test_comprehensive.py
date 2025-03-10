#!/usr/bin/env python3
"""
Comprehensive test script for the medical assistant bot.
Tests various scenarios including:
- Normal symptom collection and diagnosis
- Out-of-scope message handling
- Emergency detection and response
- Follow-up question counting
"""

import asyncio
import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the bot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_bot import get_bot_instance

# Test scenarios
class TestScenario:
    def __init__(self, name: str, messages: List[str], expected_states: List[str] = None):
        self.name = name
        self.messages = messages
        self.expected_states = expected_states or []
        self.responses = []
        self.states = []
        self.passed = False

    def __str__(self):
        return f"Scenario: {self.name} - {'PASSED' if self.passed else 'FAILED'}"

async def run_test_scenario(bot, scenario: TestScenario, user_id: str = "test_user") -> None:
    """
    Run a test scenario and collect the responses and states.
    """
    logger.info(f"\n{'='*80}\nRunning scenario: {scenario.name}\n{'='*80}")
    
    # Reset user state
    bot.dialog_manager.reset_user_state(user_id)
    
    # Process each message
    for i, message in enumerate(scenario.messages):
        logger.info(f"\nUser message [{i+1}/{len(scenario.messages)}]: {message}")
        response = await bot.process_message(user_id, message)
        state = bot.dialog_manager.get_user_state(user_id)
        
        scenario.responses.append(response)
        scenario.states.append(state)
        
        logger.info(f"Bot response: {response}")
        logger.info(f"Current state: {state}")
        
        # Check if we've reached an emergency state - if so, we can stop
        if state == "emergency":
            logger.info("Emergency state detected - stopping scenario")
            break
    
    # Check if expected states match actual states
    if scenario.expected_states:
        states_match = len(scenario.expected_states) <= len(scenario.states)
        if states_match:
            for i, expected_state in enumerate(scenario.expected_states):
                if expected_state != scenario.states[i]:
                    states_match = False
                    break
        scenario.passed = states_match
        
        if not states_match:
            logger.error(f"State mismatch in scenario '{scenario.name}'")
            logger.error(f"Expected states: {scenario.expected_states}")
            logger.error(f"Actual states: {scenario.states[:len(scenario.expected_states)]}")
    else:
        # If no expected states, just mark as passed
        scenario.passed = True
    
    return scenario

async def main():
    # Initialize the bot
    bot = await get_bot_instance()
    
    # Define test scenarios
    scenarios = [
        # 1. Normal symptom collection and diagnosis
        TestScenario(
            name="Normal Symptom Collection and Diagnosis",
            messages=[
                "Hello",
                "I have a headache and fever",
                "Yes, I also feel tired",
                "My temperature is 101F",
                "I've had it for 2 days",
                "Yes, that sounds right",
            ],
            expected_states=[
                "greeting",
                "collecting_symptoms",
                "collecting_symptoms",
                "collecting_symptoms",
                "verification",
                "generating_diagnosis"
            ]
        ),
        
        # 2. Out-of-scope handling during symptom collection
        TestScenario(
            name="Out-of-Scope Handling with Re-asking Questions",
            messages=[
                "Hello",
                "I have a sore throat",
                "What's the weather like today?",  # Out-of-scope - should re-ask about symptoms
                "My throat hurts when I swallow",
                "Yes, and I have a slight fever"
            ],
            expected_states=[
                "greeting",
                "collecting_symptoms",
                "out_of_scope_handler",
                "collecting_symptoms",
                "collecting_symptoms"
            ]
        ),
        
        # 3. Emergency detection
        TestScenario(
            name="Emergency Detection",
            messages=[
                "Hello",
                "I'm having severe chest pain and difficulty breathing",  # Emergency symptoms
                "It feels like someone is sitting on my chest",  # Should stay in emergency state
                "Should I take aspirin?"  # Should stay in emergency state
            ],
            expected_states=[
                "greeting",
                "emergency",
                "emergency",
                "emergency"
            ]
        ),
        
        # 4. Follow-up question counting (only symptom-related)
        TestScenario(
            name="Follow-up Question Counting",
            messages=[
                "Hello",
                "I have a cough",
                "What's your name?",  # Out-of-scope, shouldn't count
                "It's a dry cough",
                "What's the capital of France?",  # Out-of-scope, shouldn't count
                "I've had it for a week",
                "Yes, I also have a sore throat",
                "It hurts when I swallow",  # This should be the 4th symptom-related answer
            ],
            expected_states=[
                "greeting",
                "collecting_symptoms",
                "out_of_scope_handler",
                "collecting_symptoms",
                "out_of_scope_handler",
                "collecting_symptoms",
                "collecting_symptoms",
                "verification"  # Should transition after 4 symptom-related questions
            ]
        ),
        
        # 5. Mixed scenario with multiple intents
        TestScenario(
            name="Mixed Scenario",
            messages=[
                "Hello",
                "What causes migraines?",  # Medical inquiry
                "I actually have a migraine right now",  # Symptom reporting
                "It's on the right side of my head",
                "What's your favorite color?",  # Out-of-scope
                "The pain is throbbing",
                "I feel nauseous too",
                "Yes, I've had migraines before"
            ]
        )
    ]
    
    # Run all scenarios
    results = []
    for scenario in scenarios:
        result = await run_test_scenario(bot, scenario)
        results.append(result)
    
    # Print summary
    logger.info("\n\n" + "="*80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    passed = 0
    for result in results:
        status = "PASSED" if result.passed else "FAILED"
        logger.info(f"{result.name}: {status}")
        if result.passed:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(results)} scenarios")

if __name__ == "__main__":
    asyncio.run(main())
