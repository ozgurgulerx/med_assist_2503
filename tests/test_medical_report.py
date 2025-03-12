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

async def test_high_confidence_report():
    """Test the formal medical report format with high confidence case."""
    bot = MedicalAssistantBot()
    user_id = "test_high_conf"
    
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
    
    # Add symptoms for migraine with aura (high confidence case)
    symptoms = [
        "severe headache",
        "sensitivity to light",
        "nausea",
        "visual disturbances",
        "throbbing pain on one side of head"
    ]
    
    print("\n===== Adding symptoms for HIGH CONFIDENCE case =====")
    for symptom in symptoms:
        print(f"Adding symptom: {symptom}")
        bot.diagnostic_engine.add_symptom(patient_data, symptom)
    
    # Update diagnosis confidence
    await bot.diagnostic_engine.update_diagnosis_confidence(patient_data)
    
    # Add a diagnosis name and high confidence
    if "diagnosis" not in patient_data:
        patient_data["diagnosis"] = {}
    patient_data["diagnosis"]["name"] = "Migraine with Aura"
    patient_data["diagnosis"]["confidence"] = 0.88
    patient_data["confidence_reasoning"] = "The combination of severe headache, sensitivity to light, nausea, visual disturbances, and throbbing pain on one side of the head is highly indicative of migraine with aura. The pattern, severity, and constellation of symptoms strongly align with established diagnostic criteria for this condition."
    
    # Add follow-up questions with answers
    if "asked_questions" not in patient_data:
        patient_data["asked_questions"] = []
    patient_data["asked_questions"].extend([
        {
            "question": "How long have you been experiencing these headaches?",
            "answer": "About 3 years, they come and go every few weeks",
            "timestamp_asked": "2025-03-10T10:30:00Z",
            "timestamp_answered": "2025-03-10T10:30:15Z"
        },
        {
            "question": "Do you have a family history of migraines?",
            "answer": "Yes, my mother had similar headaches",
            "timestamp_asked": "2025-03-10T10:31:00Z",
            "timestamp_answered": "2025-03-10T10:31:10Z"
        },
        {
            "question": "Have you identified any triggers for these episodes?",
            "answer": "Stress and lack of sleep seem to make them worse",
            "timestamp_asked": "2025-03-10T10:32:00Z",
            "timestamp_answered": "2025-03-10T10:32:20Z"
        },
        {
            "question": "On a scale of 1-10, how would you rate the pain intensity?",
            "answer": "Usually around 8 when they're bad",
            "timestamp_asked": "2025-03-10T10:33:00Z",
            "timestamp_answered": "2025-03-10T10:33:05Z"
        }
    ])
    
    # Add verification info for high confidence
    if "verification_info" not in patient_data:
        patient_data["verification_info"] = {}
    patient_data["verification_info"]["trigger_reason"] = "high_confidence"
    
    # Set the dialog state to verification
    bot.dialog_manager.set_user_state(user_id, "verification")
    
    # Add model information
    patient_data["diagnosis_model"] = "GPT-4/medical-diagnostic-2025"
    patient_data["verification_model"] = "O1/medical-verification"
    
    # Process a test message to generate a report
    response = await bot.process_message(user_id, "Is this a serious condition?", include_diagnostics=True)
    
    # Print the response with the diagnostic report
    print("\n\n===== HIGH CONFIDENCE MEDICAL REPORT =====")
    print(response)

async def test_low_confidence_report():
    """Test the formal medical report format with low confidence case."""
    bot = MedicalAssistantBot()
    user_id = "test_low_conf"
    
    # First, let's get the patient data structure
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    
    # Add some demographics
    if "demographics" not in patient_data:
        patient_data["demographics"] = {}
    patient_data["demographics"]["name"] = "Jane Smith"
    patient_data["demographics"]["age"] = "38"
    patient_data["demographics"]["gender"] = "Female"
    patient_data["demographics"]["height"] = "165 cm"
    patient_data["demographics"]["weight"] = "62 kg"
    
    # Add ambiguous symptoms (low confidence case)
    symptoms = [
        "fatigue",
        "occasional dizziness",
        "mild joint pain",
        "intermittent headache"
    ]
    
    print("\n===== Adding symptoms for LOW CONFIDENCE case =====")
    for symptom in symptoms:
        print(f"Adding symptom: {symptom}")
        bot.diagnostic_engine.add_symptom(patient_data, symptom)
    
    # Update diagnosis confidence
    await bot.diagnostic_engine.update_diagnosis_confidence(patient_data)
    
    # Add a diagnosis name and low confidence
    if "diagnosis" not in patient_data:
        patient_data["diagnosis"] = {}
    patient_data["diagnosis"]["name"] = "Unspecified Condition"
    patient_data["diagnosis"]["confidence"] = 0.45
    patient_data["confidence_reasoning"] = "The symptoms of fatigue, occasional dizziness, mild joint pain, and intermittent headache are non-specific and could be associated with numerous conditions ranging from stress and lack of sleep to more serious underlying medical issues. Without more specific symptoms or diagnostic markers, it is difficult to narrow down to a specific diagnosis with high confidence."
    
    # Add follow-up questions with answers
    if "asked_questions" not in patient_data:
        patient_data["asked_questions"] = []
    patient_data["asked_questions"].extend([
        {
            "question": "How long have you been experiencing these symptoms?",
            "answer": "The fatigue has been ongoing for about 2 months, other symptoms come and go",
            "timestamp_asked": "2025-03-10T11:30:00Z",
            "timestamp_answered": "2025-03-10T11:30:20Z"
        },
        {
            "question": "Have you noticed any patterns to when these symptoms occur?",
            "answer": "Not really, they seem random but maybe worse when I'm stressed",
            "timestamp_asked": "2025-03-10T11:31:00Z",
            "timestamp_answered": "2025-03-10T11:31:15Z"
        },
        {
            "question": "Have you had any recent changes in medication or diet?",
            "answer": "No changes in medication, but I have been trying to eat healthier",
            "timestamp_asked": "2025-03-10T11:32:00Z",
            "timestamp_answered": "2025-03-10T11:32:10Z"
        },
        {
            "question": "Have you had any recent illnesses or infections?",
            "answer": "I had a mild cold about a month ago, but it cleared up quickly",
            "timestamp_asked": "2025-03-10T11:33:00Z",
            "timestamp_answered": "2025-03-10T11:33:25Z"
        }
    ])
    
    # Add verification info for low confidence
    if "verification_info" not in patient_data:
        patient_data["verification_info"] = {}
    patient_data["verification_info"]["trigger_reason"] = "low_confidence"
    
    # Add low confidence explanation from O1 model
    patient_data["verification_info"]["low_confidence_explanation"] = """Your symptoms of fatigue, occasional dizziness, mild joint pain, and intermittent headache present a diagnostic challenge as they don't clearly align with a single medical condition with high confidence. These symptoms are non-specific and could be associated with various conditions including chronic fatigue syndrome, fibromyalgia, early autoimmune disorders, vitamin deficiencies, or even stress-related conditions.

Possible conditions to consider include:
• Chronic Fatigue Syndrome
• Fibromyalgia
• Early stages of an autoimmune disorder like rheumatoid arthritis or lupus
• Vitamin D or B12 deficiency
• Thyroid dysfunction
• Stress-related conditions
• Sleep disorders

I strongly recommend consulting with a primary care physician who can conduct a comprehensive evaluation, including a thorough physical examination and appropriate laboratory tests. They may refer you to specialists such as a rheumatologist, endocrinologist, or neurologist depending on their initial findings.

While your symptoms don't appear immediately life-threatening, please seek prompt medical attention if you develop any of these warning signs: severe headache with neck stiffness, sudden severe dizziness with loss of balance, unexplained weight loss, persistent fever, or worsening joint pain with swelling and redness."""
    
    # Add referral needed flag
    patient_data["referral_needed"] = True
    
    # Set the dialog state to verification
    bot.dialog_manager.set_user_state(user_id, "verification")
    
    # Add model information
    patient_data["diagnosis_model"] = "GPT-4/medical-diagnostic-2025"
    patient_data["verification_model"] = "O1/medical-verification"
    
    # Process a test message to generate a report
    response = await bot.process_message(user_id, "What could be causing my symptoms?", include_diagnostics=True)
    
    # Print the response with the diagnostic report
    print("\n\n===== LOW CONFIDENCE MEDICAL REPORT =====")
    print(response)

async def main():
    """Run both test cases."""
    await test_high_confidence_report()
    await test_low_confidence_report()

if __name__ == "__main__":
    asyncio.run(main())
