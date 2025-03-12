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

async def test_symptom_extraction():
    """Test the symptom extraction functionality of the medical assistant bot."""
    bot = MedicalAssistantBot()
    user_id = "test_user"
    
    # Test cases with different symptom reporting formats
    test_cases = [
        "I have a headache",
        "I'm feeling nauseous",
        "I've been experiencing fever and chills",
        "My throat is sore",
        "I'm having difficulty breathing"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n\n===== Test Case {i+1}: '{test_case}' =====")
        
        # Process the message
        response = await bot.process_message(user_id, test_case, include_diagnostics=True)
        
        # Print the response
        print(f"Bot response: {response}")
        
        # Get the user data to check symptoms and diagnosis confidence
        user_data = bot.get_user_data(user_id)
        patient_data = user_data.get("patient_data", {})
        
        # Print the symptoms and diagnosis confidence
        symptoms = patient_data.get("symptoms", [])
        diagnosis = patient_data.get("diagnosis", {})
        confidence = diagnosis.get("confidence", 0.0)
        confidence_reasoning = patient_data.get("confidence_reasoning", "")
        
        print(f"Extracted symptoms: {symptoms}")
        print(f"Diagnosis confidence: {confidence:.2f}")
        print(f"Confidence reasoning: {confidence_reasoning}")

if __name__ == "__main__":
    asyncio.run(test_symptom_extraction())
