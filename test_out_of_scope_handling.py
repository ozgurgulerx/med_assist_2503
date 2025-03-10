import asyncio
import logging
from core_bot import MedicalAssistantBot

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

async def test_out_of_scope_handling():
    """Test how the bot handles out-of-scope intents during symptom collection."""
    bot = MedicalAssistantBot()
    user_id = "test_out_of_scope_user"
    
    print("\n===== TESTING OUT-OF-SCOPE HANDLING =====\n")
    
    # Start with a greeting
    print("\n1. Starting conversation with a greeting")
    response = await bot.process_message(user_id, "Hello, I need some medical help")
    print(f"Bot: {response}")
    
    # Report initial symptoms
    print("\n2. Reporting initial symptoms")
    response = await bot.process_message(user_id, "I have a severe headache and nausea")
    print(f"Bot: {response}")
    
    # Inject an out-of-scope message during symptom collection
    print("\n3. Injecting out-of-scope message during symptom collection")
    response = await bot.process_message(user_id, "What's the weather like today?")
    print(f"Bot: {response}")
    
    # Check if the bot steers back to symptom collection
    print("\n4. Checking if bot steers back to symptom collection")
    response = await bot.process_message(user_id, "Yes, back to my symptoms")
    print(f"Bot: {response}")
    
    # Add more symptoms
    print("\n5. Adding more symptoms")
    response = await bot.process_message(user_id, "I also have sensitivity to light")
    print(f"Bot: {response}")
    
    # Inject another out-of-scope message
    print("\n6. Injecting another out-of-scope message")
    response = await bot.process_message(user_id, "Can you tell me about the latest news?")
    print(f"Bot: {response}")
    
    # Continue with symptom reporting
    print("\n7. Continuing with symptom reporting")
    response = await bot.process_message(user_id, "I've been having these headaches for 3 days")
    print(f"Bot: {response}")
    
    # Check the current state and diagnosis progress
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    current_state = bot.dialog_manager.get_user_state(user_id)
    
    print("\n===== CONVERSATION STATE SUMMARY =====")
    print(f"Current dialog state: {current_state}")
    print(f"Collected symptoms: {[s['name'] if isinstance(s, dict) else s for s in patient_data.get('symptoms', [])]}")
    print(f"Diagnosis confidence: {patient_data.get('diagnosis', {}).get('confidence', 0.0):.2f}")
    print(f"Asked questions: {len(patient_data.get('asked_questions', []))}")

async def test_out_of_scope_during_verification():
    """Test how the bot handles out-of-scope intents during verification phase."""
    bot = MedicalAssistantBot()
    user_id = "test_verification_oos_user"
    
    # First, let's get the patient data structure
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    
    # Add symptoms for migraine
    symptoms = [
        "severe headache",
        "sensitivity to light",
        "nausea",
        "visual disturbances"
    ]
    
    print("\n===== TESTING OUT-OF-SCOPE DURING VERIFICATION =====\n")
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
    print(f"Dialog state set to: {bot.dialog_manager.get_user_state(user_id)}")
    
    # Send a message to trigger verification
    print("\n2. Triggering verification phase")
    response = await bot.process_message(user_id, "Are these all my symptoms?")
    print(f"Bot: {response}")
    
    # Inject an out-of-scope message during verification
    print("\n3. Injecting out-of-scope message during verification")
    response = await bot.process_message(user_id, "What's your favorite movie?")
    print(f"Bot: {response}")
    
    # Check if the bot steers back to verification
    print("\n4. Checking if bot steers back to verification")
    response = await bot.process_message(user_id, "Yes, let's continue with my diagnosis")
    print(f"Bot: {response}")
    
    # Check the current state
    current_state = bot.dialog_manager.get_user_state(user_id)
    print(f"\nFinal dialog state: {current_state}")

async def test_out_of_scope_during_diagnosis():
    """Test how the bot handles out-of-scope intents during diagnosis generation."""
    bot = MedicalAssistantBot()
    user_id = "test_diagnosis_oos_user"
    
    # First, let's get the patient data structure
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    
    # Add symptoms for common cold
    symptoms = [
        "runny nose",
        "sore throat",
        "cough",
        "mild fever"
    ]
    
    print("\n===== TESTING OUT-OF-SCOPE DURING DIAGNOSIS =====\n")
    print("1. Setting up patient with cold symptoms")
    
    for symptom in symptoms:
        bot.diagnostic_engine.add_symptom(patient_data, symptom)
    
    # Update diagnosis confidence
    await bot.diagnostic_engine.update_diagnosis_confidence(patient_data)
    
    # Set diagnosis info
    if "diagnosis" not in patient_data:
        patient_data["diagnosis"] = {}
    patient_data["diagnosis"]["name"] = "Common Cold"
    patient_data["diagnosis"]["confidence"] = 0.90
    
    # Set the dialog state to generating_diagnosis
    bot.dialog_manager.set_user_state(user_id, "generating_diagnosis")
    print(f"Dialog state set to: {bot.dialog_manager.get_user_state(user_id)}")
    
    # Send a message to trigger diagnosis generation
    print("\n2. Triggering diagnosis generation")
    response = await bot.process_message(user_id, "What's wrong with me?")
    print(f"Bot: {response}")
    
    # Inject an out-of-scope message during diagnosis
    print("\n3. Injecting out-of-scope message during diagnosis")
    response = await bot.process_message(user_id, "Can you recommend a good restaurant?")
    print(f"Bot: {response}")
    
    # Check if the bot handles it appropriately
    print("\n4. Checking if bot handles out-of-scope appropriately")
    response = await bot.process_message(user_id, "Thanks, now back to my health")
    print(f"Bot: {response}")
    
    # Check the current state
    current_state = bot.dialog_manager.get_user_state(user_id)
    print(f"\nFinal dialog state: {current_state}")

async def main():
    """Run all test cases."""
    await test_out_of_scope_handling()
    await test_out_of_scope_during_verification()
    await test_out_of_scope_during_diagnosis()

if __name__ == "__main__":
    asyncio.run(main())
