#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Testing Script for Medical Assistant Bot

This script tests the major flows of the Medical Assistant application:
1. Diagnostic flow - Testing if the bot can generate a diagnosis
2. Follow-up questions flow - Testing if the bot asks relevant follow-up questions
3. Emergency detection flow - Testing if the bot identifies urgent situations
4. General information flow - Testing if the bot provides medical information

Usage:
    python automated_testing.py
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Dict, Any, List, Tuple
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_results.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER_ID = "automated_test_user"
TIMEOUT = 60  # seconds


def print_header(message: str) -> None:
    """Print a formatted header for test sections"""
    print(f"\n{Fore.CYAN}{'='*80}\n{message}\n{'='*80}{Style.RESET_ALL}\n")


def print_success(message: str) -> None:
    """Print a success message"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message: str) -> None:
    """Print an error message"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_info(message: str) -> None:
    """Print an info message"""
    print(f"{Fore.YELLOW}ℹ {message}{Style.RESET_ALL}")


def check_server_health() -> bool:
    """Check if the server is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_info(f"Server health: {data.get('status', 'unknown')}")
            print_info(f"Connection status: {data.get('connection', 'unknown')}")
            print_info(f"Active users: {data.get('active_users', 0)}")
            return data.get('connection') == 'healthy'
        return False
    except requests.RequestException as e:
        print_error(f"Server health check failed: {str(e)}")
        return False


def reset_conversation() -> bool:
    """Reset the conversation for the test user"""
    try:
        response = requests.delete(f"{API_BASE_URL}/users/{TEST_USER_ID}", timeout=TIMEOUT)
        return response.status_code == 200
    except requests.RequestException as e:
        print_error(f"Failed to reset conversation: {str(e)}")
        return False


def send_message(message: str, include_diagnostics: bool = True) -> Dict[str, Any]:
    """Send a message to the bot and return the response"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "user_id": TEST_USER_ID,
                "include_diagnostics": include_diagnostics
            },
            timeout=TIMEOUT
        )
        
        if response.status_code != 200:
            print_error(f"Failed to send message: Status code {response.status_code}")
            print_error(response.text)
            return {}
        
        return response.json()
    except requests.RequestException as e:
        print_error(f"Request failed: {str(e)}")
        return {}


def test_diagnosis_flow() -> bool:
    """Test the diagnosis generation flow"""
    print_header("Testing Diagnosis Flow")
    reset_conversation()
    
    # Test with a common migraine case
    messages = [
        "I've been having a throbbing headache on one side of my head for the past 3 days. "
        "It gets worse with light and noise, and sometimes I feel nauseous."
    ]
    
    for i, message in enumerate(messages):
        print_info(f"User: {message}")
        response = send_message(message)
        
        if not response:
            print_error("No response received")
            return False
            
        bot_response = response.get("response", "")
        print_info(f"Bot: {bot_response[:100]}..." if len(bot_response) > 100 else f"Bot: {bot_response}")
        
        # Check for diagnosis indicators in the response
        if "diagnosis" in bot_response.lower() or "assessment" in bot_response.lower() or "migraine" in bot_response.lower():
            print_success("Diagnosis generated successfully")
            return True
        
        time.sleep(1)  # Pause between messages
    
    print_error("No diagnosis was generated")
    return False


def test_followup_questions_flow() -> bool:
    """Test if the bot asks appropriate follow-up questions"""
    print_header("Testing Follow-up Questions Flow")
    reset_conversation()
    
    # Start with a vague symptom to trigger follow-up questions
    initial_message = "I have a pain in my stomach"
    print_info(f"User: {initial_message}")
    
    response = send_message(initial_message)
    if not response:
        print_error("No response received")
        return False
        
    bot_response = response.get("response", "")
    print_info(f"Bot: {bot_response[:100]}..." if len(bot_response) > 100 else f"Bot: {bot_response}")
    
    # Check if the response contains a question
    if "?" in bot_response:
        print_success("Bot asked a follow-up question")
        
        # Provide more detailed information
        followup_answer = "The pain is in the lower right side of my abdomen. It started yesterday and gets worse when I move."
        print_info(f"User: {followup_answer}")
        
        response = send_message(followup_answer)
        if not response:
            print_error("No response received for follow-up")
            return False
            
        bot_response = response.get("response", "")
        print_info(f"Bot: {bot_response[:100]}..." if len(bot_response) > 100 else f"Bot: {bot_response}")
        
        # Check if the bot provided a more specific response after the follow-up
        if "appendicitis" in bot_response.lower() or "diagnosis" in bot_response.lower() or "assessment" in bot_response.lower():
            print_success("Bot provided more specific information after follow-up")
            return True
    
    print_error("Bot did not ask follow-up questions or provide specific information")
    return False


def test_emergency_detection() -> bool:
    """Test if the bot correctly identifies emergency situations"""
    print_header("Testing Emergency Detection Flow")
    reset_conversation()
    
    # Describe clear emergency symptoms
    emergency_message = "I'm having severe chest pain radiating to my left arm and jaw. I'm short of breath and feeling dizzy."
    print_info(f"User: {emergency_message}")
    
    response = send_message(emergency_message)
    if not response:
        print_error("No response received")
        return False
        
    bot_response = response.get("response", "")
    print_info(f"Bot: {bot_response[:100]}..." if len(bot_response) > 100 else f"Bot: {bot_response}")
    
    # Check if the response contains emergency keywords
    emergency_keywords = ["emergency", "immediate", "urgent", "call 911", "ambulance", "hospital"]
    
    for keyword in emergency_keywords:
        if keyword in bot_response.lower():
            print_success(f"Bot correctly identified emergency situation (keyword: {keyword})")
            return True
    
    print_error("Bot did not identify emergency situation")
    return False


def test_medical_information() -> bool:
    """Test if the bot provides general medical information"""
    print_header("Testing Medical Information Flow")
    reset_conversation()
    
    # Ask for general medical information
    info_message = "What are the symptoms of diabetes?"
    print_info(f"User: {info_message}")
    
    response = send_message(info_message)
    if not response:
        print_error("No response received")
        return False
        
    bot_response = response.get("response", "")
    print_info(f"Bot: {bot_response[:100]}..." if len(bot_response) > 100 else f"Bot: {bot_response}")
    
    # Check if the response contains relevant information
    diabetes_keywords = ["diabetes", "thirst", "urination", "hunger", "fatigue", "blurred", "glucose", "sugar"]
    
    keyword_matches = [keyword for keyword in diabetes_keywords if keyword in bot_response.lower()]
    if keyword_matches:
        print_success(f"Bot provided relevant information about diabetes (keywords: {', '.join(keyword_matches)})")
        return True
    
    print_error("Bot did not provide relevant information about diabetes")
    return False


def test_error_handling() -> bool:
    """Test the bot's error handling capabilities"""
    print_header("Testing Error Handling")
    reset_conversation()
    
    # Send a very long message to potentially trigger processing issues
    long_message = "I have a symptom " * 1000
    print_info(f"User: {long_message[:50]}... (very long message)")
    
    response = send_message(long_message)
    
    # Bot should respond with something, even if it's an error message
    if response and "response" in response:
        print_success("Bot handled extremely long input without crashing")
        return True
    
    print_error("Bot failed to handle the extreme input")
    return False


def run_all_tests() -> Dict[str, bool]:
    """Run all test cases and return results"""
    results = {}
    
    # Check if server is running
    if not check_server_health():
        print_error("Server is not running or not healthy. Aborting tests.")
        return {"server_health": False}
    
    results["server_health"] = True
    
    # Run all tests and collect results
    test_functions = [
        ("diagnosis_flow", test_diagnosis_flow),
        ("followup_questions", test_followup_questions_flow),
        ("emergency_detection", test_emergency_detection),
        ("medical_information", test_medical_information),
        ("error_handling", test_error_handling),
    ]
    
    for name, func in test_functions:
        try:
            results[name] = func()
        except Exception as e:
            print_error(f"Test {name} failed with exception: {str(e)}")
            results[name] = False
    
    # Print summary
    print_header("Test Results Summary")
    for name, result in results.items():
        if result:
            print_success(f"{name}: Passed")
        else:
            print_error(f"{name}: Failed")
    
    # Calculate overall success rate
    success_rate = sum(1 for result in results.values() if result) / len(results) * 100
    print_info(f"Overall success rate: {success_rate:.1f}%")
    
    return results


if __name__ == "__main__":
    print_header("Starting Automated Tests for Medical Assistant Bot")
    results = run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
