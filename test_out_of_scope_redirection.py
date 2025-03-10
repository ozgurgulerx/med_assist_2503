import asyncio
import logging
import json
from core_bot import MedicalAssistantBot

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

async def simulate_conversation(bot, user_id, messages, print_state=True):
    """Simulate a conversation with the bot and track state transitions."""
    states = []
    responses = []
    
    for i, message in enumerate(messages):
        print(f"\nUser: {message}")
        response = await bot.process_message(user_id, message)
        print(f"Bot: {response}")
        
        # Get current state after processing
        current_state = bot.dialog_manager.get_user_state(user_id)
        states.append(current_state)
        responses.append(response)
        
        if print_state:
            print(f"[Current state: {current_state}]")
    
    return states, responses

async def test_out_of_scope_during_symptom_collection():
    """Test how the bot handles out-of-scope messages during symptom collection."""
    bot = MedicalAssistantBot()
    user_id = "test_oos_symptoms_user"
    
    print("\n===== TEST: OUT-OF-SCOPE DURING SYMPTOM COLLECTION =====\n")
    
    # Define a conversation with out-of-scope messages
    conversation = [
        "Hello, I need some medical advice",  # greeting
        "I've been having headaches and dizziness",  # symptom reporting
        "What's the weather like today?",  # out-of-scope
        "Sorry, back to my symptoms. I also have blurry vision",  # symptom reporting
        "Can you recommend a good movie to watch?",  # out-of-scope
        "My headaches get worse when I read"  # symptom reporting
    ]
    
    # Simulate the conversation
    states, _ = await simulate_conversation(bot, user_id, conversation)
    
    # Analyze state transitions
    print("\nState transitions:")
    for i, (message, state) in enumerate(zip(conversation, states)):
        print(f"{i+1}. '{message}' → {state}")
    
    # Check if out-of-scope messages were handled correctly
    out_of_scope_indices = [2, 4]  # Indices of out-of-scope messages
    for idx in out_of_scope_indices:
        if states[idx] == "out_of_scope_handler":
            print(f"✓ Message '{conversation[idx]}' correctly identified as out-of-scope")
        else:
            print(f"✗ Message '{conversation[idx]}' not identified as out-of-scope (state: {states[idx]})")
    
    # Check if the conversation returned to symptom collection after out-of-scope
    if states[-1] == "collecting_symptoms":
        print("✓ Conversation successfully returned to symptom collection after out-of-scope messages")
    else:
        print(f"✗ Conversation did not return to symptom collection (final state: {states[-1]})")

async def test_out_of_scope_during_verification():
    """Test how the bot handles out-of-scope messages during verification phase."""
    bot = MedicalAssistantBot()
    user_id = "test_oos_verification_user"
    
    # Set up patient with symptoms and trigger verification
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    
    # Add symptoms for migraine
    symptoms = [
        "severe headache",
        "sensitivity to light",
        "nausea",
        "visual disturbances"
    ]
    
    print("\n===== TEST: OUT-OF-SCOPE DURING VERIFICATION =====\n")
    print("1. Setting up patient with migraine symptoms")
    
    for symptom in symptoms:
        bot.diagnostic_engine.add_symptom(patient_data, symptom)
    
    # Update diagnosis confidence to trigger verification
    await bot.diagnostic_engine.update_diagnosis_confidence(patient_data)
    
    # Manually set high confidence to trigger verification
    if "diagnosis" not in patient_data:
        patient_data["diagnosis"] = {}
    patient_data["diagnosis"]["name"] = "Migraine with Aura"
    patient_data["diagnosis"]["confidence"] = 0.88
    
    # Add verification info
    if "verification_info" not in patient_data:
        patient_data["verification_info"] = {}
    patient_data["verification_info"]["trigger_reason"] = "high_confidence"
    
    # Set the dialog state to verification
    bot.dialog_manager.set_user_state(user_id, "verification")
    print(f"Initial dialog state set to: {bot.dialog_manager.get_user_state(user_id)}")
    
    # Define a conversation with out-of-scope messages during verification
    conversation = [
        "Are these all my symptoms?",  # verification
        "What's your favorite movie?",  # out-of-scope
        "Let's continue with my diagnosis",  # back to verification
        "Yes, those are all my symptoms"  # verification confirmation
    ]
    
    # Simulate the conversation
    states, _ = await simulate_conversation(bot, user_id, conversation)
    
    # Analyze state transitions
    print("\nState transitions:")
    for i, (message, state) in enumerate(zip(conversation, states)):
        print(f"{i+1}. '{message}' → {state}")
    
    # Check if out-of-scope message was handled correctly
    if states[1] == "out_of_scope_handler":
        print(f"✓ Message '{conversation[1]}' correctly identified as out-of-scope")
    else:
        print(f"✗ Message '{conversation[1]}' not identified as out-of-scope (state: {states[1]})")
    
    # Check if the conversation returned to verification after out-of-scope
    if states[2] == "verification":
        print("✓ Conversation successfully returned to verification after out-of-scope message")
    else:
        print(f"✗ Conversation did not return to verification (state after out-of-scope: {states[2]})")

async def test_multiple_consecutive_out_of_scope():
    """Test how the bot handles multiple consecutive out-of-scope messages."""
    bot = MedicalAssistantBot()
    user_id = "test_multiple_oos_user"
    
    print("\n===== TEST: MULTIPLE CONSECUTIVE OUT-OF-SCOPE MESSAGES =====\n")
    
    # Define a conversation with multiple consecutive out-of-scope messages
    conversation = [
        "Hello, I have a medical question",  # greeting
        "I've been feeling very tired lately",  # symptom reporting
        "What's the weather forecast?",  # out-of-scope
        "Can you tell me a joke?",  # out-of-scope
        "What's the capital of France?",  # out-of-scope
        "Sorry, back to my symptoms. I also have joint pain"  # symptom reporting
    ]
    
    # Simulate the conversation
    states, responses = await simulate_conversation(bot, user_id, conversation)
    
    # Analyze state transitions
    print("\nState transitions:")
    for i, (message, state) in enumerate(zip(conversation, states)):
        print(f"{i+1}. '{message}' → {state}")
    
    # Check if out-of-scope messages were handled correctly
    out_of_scope_indices = [2, 3, 4]  # Indices of out-of-scope messages
    for idx in out_of_scope_indices:
        if states[idx] == "out_of_scope_handler":
            print(f"✓ Message '{conversation[idx]}' correctly identified as out-of-scope")
        else:
            print(f"✗ Message '{conversation[idx]}' not identified as out-of-scope (state: {states[idx]})")
    
    # Check if the conversation returned to symptom collection after multiple out-of-scope messages
    if states[-1] == "collecting_symptoms":
        print("✓ Conversation successfully returned to symptom collection after multiple out-of-scope messages")
    else:
        print(f"✗ Conversation did not return to symptom collection (final state: {states[-1]})")
    
    # Check if the redirection messages were appropriate
    for idx in out_of_scope_indices:
        if "I understand, but" in responses[idx] and "medical" in responses[idx].lower():
            print(f"✓ Appropriate redirection message for out-of-scope message {idx+1}")
        else:
            print(f"✗ Redirection message for out-of-scope message {idx+1} may not be appropriate")

async def main():
    """Run all test cases."""
    await test_out_of_scope_during_symptom_collection()
    await test_out_of_scope_during_verification()
    await test_multiple_consecutive_out_of_scope()

if __name__ == "__main__":
    asyncio.run(main())
