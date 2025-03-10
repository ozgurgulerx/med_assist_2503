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

async def test_medical_report_format():
    """Test the new formal medical report format."""
    bot = MedicalAssistantBot()
    user_id = "test_report_user"
    
    # First, let's get the patient data structure
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    
    # Add some demographics
    if "demographics" not in patient_data:
        patient_data["demographics"] = {}
    patient_data["demographics"]["name"] = "John Doe"
    patient_data["demographics"]["age"] = "45"
    patient_data["demographics"]["gender"] = "Male"
    patient_data["demographics"]["height"] = "180 cm"
    patient_data["demographics"]["weight"] = "75 kg"
    
    # Add some symptoms
    symptoms = [
        "severe headache",
        "sensitivity to light",
        "nausea",
        "visual disturbances"
    ]
    
    print("\n===== Adding symptoms to patient data =====")
    for symptom in symptoms:
        print(f"Adding symptom: {symptom}")
        bot.diagnostic_engine.add_symptom(patient_data, symptom)
    
    # Update diagnosis confidence
    await bot.diagnostic_engine.update_diagnosis_confidence(patient_data)
    
    # Add a diagnosis name
    if "diagnosis" not in patient_data:
        patient_data["diagnosis"] = {}
    patient_data["diagnosis"]["name"] = "Migraine with Aura"
    patient_data["diagnosis"]["confidence"] = 0.85
    patient_data["confidence_reasoning"] = "The combination of severe headache, sensitivity to light, nausea, and visual disturbances is highly indicative of migraine with aura. The pattern and severity of symptoms align with diagnostic criteria."
    
    # Add some follow-up questions
    if "asked_questions" not in patient_data:
        patient_data["asked_questions"] = []
    patient_data["asked_questions"].extend([
        "How long have you been experiencing these headaches?",
        "Do you have a family history of migraines?",
        "Have you identified any triggers for these episodes?",
        "On a scale of 1-10, how would you rate the pain intensity?"
    ])
    
    # Set the dialog state to verification
    bot.dialog_manager.set_user_state(user_id, "verification")
    
    # Add model information
    patient_data["diagnosis_model"] = "GPT-4/medical-diagnostic-2025"
    patient_data["verification_model"] = "O1/medical-verification-2025"
    
    # Process a test message to generate a report
    response = await bot.process_message(user_id, "How serious is my condition?", include_diagnostics=True)
    
    # Print the response with the diagnostic report
    print("\n\n===== FULL RESPONSE WITH MEDICAL REPORT =====")
    print(response)

if __name__ == "__main__":
    asyncio.run(test_medical_report_format())
