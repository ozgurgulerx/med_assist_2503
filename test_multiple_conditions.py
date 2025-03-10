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

async def test_condition(bot, user_id, name, age, gender, symptoms, diagnosis_name, confidence, questions_and_answers, verification_trigger):
    """Test the medical assistant bot with a specific condition."""
    # Get patient data
    user_data = bot.get_user_data(user_id)
    patient_data = bot.diagnostic_engine.get_patient_data(user_data)
    
    # Add demographics
    if "demographics" not in patient_data:
        patient_data["demographics"] = {}
    patient_data["demographics"]["name"] = name
    patient_data["demographics"]["age"] = age
    patient_data["demographics"]["gender"] = gender
    
    # Add symptoms
    print(f"\n===== Adding symptoms for {diagnosis_name} =====")
    for symptom in symptoms:
        print(f"Adding symptom: {symptom}")
        bot.diagnostic_engine.add_symptom(patient_data, symptom)
    
    # Update diagnosis confidence
    await bot.diagnostic_engine.update_diagnosis_confidence(patient_data)
    
    # Set diagnosis and confidence
    if "diagnosis" not in patient_data:
        patient_data["diagnosis"] = {}
    patient_data["diagnosis"]["name"] = diagnosis_name
    patient_data["diagnosis"]["confidence"] = confidence
    
    # Add follow-up questions with answers
    if "asked_questions" not in patient_data:
        patient_data["asked_questions"] = []
    
    for i, (question, answer) in enumerate(questions_and_answers):
        patient_data["asked_questions"].append({
            "question": question,
            "answer": answer,
            "timestamp_asked": f"2025-03-10T{10+i:02d}:30:00Z",
            "timestamp_answered": f"2025-03-10T{10+i:02d}:30:15Z"
        })
    
    # Add verification info
    if "verification_info" not in patient_data:
        patient_data["verification_info"] = {}
    patient_data["verification_info"]["trigger_reason"] = verification_trigger
    
    # Set referral needed for low confidence cases
    if verification_trigger == "low_confidence":
        patient_data["referral_needed"] = True
        patient_data["verification_info"]["low_confidence_explanation"] = f"Your symptoms suggest possible {diagnosis_name}, but the confidence is low at {confidence:.2f}. This could be due to the non-specific nature of some symptoms or potential overlap with other conditions. We recommend consulting with a healthcare professional for a thorough evaluation."
    
    # Set the dialog state to verification
    bot.dialog_manager.set_user_state(user_id, "verification")
    
    # Add model information
    patient_data["diagnosis_model"] = "GPT-4/medical-diagnostic-2025"
    patient_data["verification_model"] = "O1/medical-verification"
    
    # Process a test message to generate a report
    response = await bot.process_message(user_id, "What should I do about this condition?", include_diagnostics=True)
    
    # Print the response with the diagnostic report
    print(f"\n\n===== {diagnosis_name.upper()} MEDICAL REPORT =====")
    print(response)

async def test_type_2_diabetes():
    """Test the bot with Type 2 Diabetes symptoms (high confidence case)."""
    bot = MedicalAssistantBot()
    user_id = "test_diabetes"
    
    symptoms = [
        "increased thirst",
        "frequent urination",
        "unexplained weight loss",
        "fatigue",
        "blurred vision"
    ]
    
    questions_and_answers = [
        ("How long have you been experiencing these symptoms?", "About 3 months now"),
        ("Do you have a family history of diabetes?", "Yes, my father has type 2 diabetes"),
        ("Have you noticed any slow-healing sores or frequent infections?", "Yes, I've had a cut on my foot that's taking a long time to heal"),
        ("What is your approximate daily water intake?", "I drink about 3-4 liters a day now, which is much more than before")
    ]
    
    await test_condition(
        bot=bot,
        user_id=user_id,
        name="Robert Johnson",
        age="52",
        gender="Male",
        symptoms=symptoms,
        diagnosis_name="Type 2 Diabetes",
        confidence=0.92,
        questions_and_answers=questions_and_answers,
        verification_trigger="high_confidence"
    )

async def test_seasonal_allergies():
    """Test the bot with Seasonal Allergies symptoms (high confidence case)."""
    bot = MedicalAssistantBot()
    user_id = "test_allergies"
    
    symptoms = [
        "sneezing",
        "runny nose",
        "itchy eyes",
        "nasal congestion",
        "watery eyes"
    ]
    
    questions_and_answers = [
        ("Do your symptoms worsen during specific seasons?", "Yes, especially in spring when pollen counts are high"),
        ("Have you tried any over-the-counter allergy medications?", "Yes, antihistamines help somewhat but don't completely relieve symptoms"),
        ("Do you have a history of asthma or eczema?", "I had mild eczema as a child"),
        ("Do your symptoms improve when you're indoors with air conditioning?", "Yes, significantly")
    ]
    
    await test_condition(
        bot=bot,
        user_id=user_id,
        name="Sarah Williams",
        age="34",
        gender="Female",
        symptoms=symptoms,
        diagnosis_name="Seasonal Allergic Rhinitis",
        confidence=0.89,
        questions_and_answers=questions_and_answers,
        verification_trigger="high_confidence"
    )

async def test_chronic_fatigue():
    """Test the bot with Chronic Fatigue symptoms (low confidence case)."""
    bot = MedicalAssistantBot()
    user_id = "test_fatigue"
    
    symptoms = [
        "persistent fatigue",
        "unrefreshing sleep",
        "mild headaches",
        "difficulty concentrating"
    ]
    
    questions_and_answers = [
        ("How long have you been experiencing this fatigue?", "For about 8 months, it doesn't improve with rest"),
        ("Have you had any recent infections or illnesses?", "I had a viral infection about a year ago"),
        ("Have you noticed any muscle or joint pain?", "Some mild muscle aches occasionally, but not severe"),
        ("Has your activity level decreased significantly?", "Yes, I used to be very active but now struggle with basic daily tasks")
    ]
    
    await test_condition(
        bot=bot,
        user_id=user_id,
        name="Michael Chen",
        age="41",
        gender="Male",
        symptoms=symptoms,
        diagnosis_name="Chronic Fatigue Syndrome",
        confidence=0.62,
        questions_and_answers=questions_and_answers,
        verification_trigger="low_confidence"
    )

async def test_nonspecific_symptoms():
    """Test the bot with very nonspecific symptoms (very low confidence case)."""
    bot = MedicalAssistantBot()
    user_id = "test_nonspecific"
    
    symptoms = [
        "occasional dizziness",
        "mild fatigue",
        "intermittent headache"
    ]
    
    questions_and_answers = [
        ("When do you typically experience these symptoms?", "They come and go, no clear pattern"),
        ("Are the symptoms affecting your daily activities?", "Not significantly, but they're annoying"),
        ("Have you made any recent lifestyle changes?", "I've been working longer hours and sleeping less"),
        ("Have you had your blood pressure checked recently?", "Yes, it was normal at my last check-up")
    ]
    
    await test_condition(
        bot=bot,
        user_id=user_id,
        name="Emily Rodriguez",
        age="29",
        gender="Female",
        symptoms=symptoms,
        diagnosis_name="Unspecified Condition",
        confidence=0.38,
        questions_and_answers=questions_and_answers,
        verification_trigger="low_confidence"
    )

async def main():
    """Run all test cases."""
    print("\n\n===== TESTING MULTIPLE MEDICAL CONDITIONS =====\n")
    
    print("\n1. Testing Type 2 Diabetes (High Confidence)")
    await test_type_2_diabetes()
    
    print("\n2. Testing Seasonal Allergies (High Confidence)")
    await test_seasonal_allergies()
    
    print("\n3. Testing Chronic Fatigue Syndrome (Low Confidence)")
    await test_chronic_fatigue()
    
    print("\n4. Testing Nonspecific Symptoms (Very Low Confidence)")
    await test_nonspecific_symptoms()

if __name__ == "__main__":
    asyncio.run(main())
