import asyncio
import logging
from core_bot import MedicalAssistantBot
from diagnostic_engine import DiagnosticEngine

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

async def test_verification_threshold():
    """Test that the verification step is triggered at 85% confidence."""
    bot = MedicalAssistantBot()
    user_id = "test_verification_user"
    
    # First, let's get the patient data structure
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    
    # Set the initial state to collecting_symptoms
    bot.dialog_manager.set_user_state(user_id, "collecting_symptoms")
    print(f"Initial dialog state: {bot.dialog_manager.get_user_state(user_id)}")
    
    # Add some symptoms to build up confidence
    print("\n\n===== Adding symptoms to build confidence =====")
    symptoms = [
        "severe headache",
        "sensitivity to light",
        "nausea",
        "visual disturbances"
    ]
    
    for symptom in symptoms:
        print(f"\nAdding symptom: {symptom}")
        bot.diagnostic_engine.add_symptom(patient_data, symptom)
        await bot.diagnostic_engine.update_diagnosis_confidence(patient_data)
        
        # Print current confidence
        confidence = patient_data.get("diagnosis", {}).get("confidence", 0.0)
        print(f"Current confidence: {confidence:.2f}")
        
        # Check if we've triggered verification
        should_verify = bot.dialog_manager.should_verify_symptoms(user_id, patient_data)
        print(f"Should verify: {should_verify}")
        print(f"Current dialog state: {bot.dialog_manager.get_user_state(user_id)}")
        
        # If verification is triggered, print the current state
        if should_verify:
            print(f"Dialog state changed to: {bot.dialog_manager.get_user_state(user_id)}")
            break
    
    # If we didn't trigger verification naturally, manually set confidence to 0.85
    # and check if verification is triggered
    if bot.dialog_manager.get_user_state(user_id) != "verification":
        print("\n\n===== Testing with manual confidence of 0.85 =====")
        # Reset state to collecting_symptoms
        bot.dialog_manager.set_user_state(user_id, "collecting_symptoms")
        print(f"Reset dialog state to: {bot.dialog_manager.get_user_state(user_id)}")
        
        if "diagnosis" not in patient_data:
            patient_data["diagnosis"] = {}
        patient_data["diagnosis"]["confidence"] = 0.85
        
        # Check if we've triggered verification
        should_verify = bot.dialog_manager.should_verify_symptoms(user_id, patient_data)
        print(f"Confidence set to 0.85")
        print(f"Should verify: {should_verify}")
        print(f"Current dialog state: {bot.dialog_manager.get_user_state(user_id)}")
    
    # Test with confidence just below threshold (0.84)
    print("\n\n===== Testing with confidence of 0.84 (below threshold) =====")
    # Reset state
    bot.dialog_manager.set_user_state(user_id, "collecting_symptoms")
    print(f"Reset dialog state to: {bot.dialog_manager.get_user_state(user_id)}")
    patient_data["diagnosis"]["confidence"] = 0.84
    
    # Check if verification is triggered (should be False)
    should_verify = bot.dialog_manager.should_verify_symptoms(user_id, patient_data)
    print(f"Confidence set to 0.84")
    print(f"Should verify: {should_verify}")
    print(f"Current dialog state: {bot.dialog_manager.get_user_state(user_id)}")

if __name__ == "__main__":
    asyncio.run(test_verification_threshold())
