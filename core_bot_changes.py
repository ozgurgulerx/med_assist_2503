# Changes to implement in core_bot.py

# 1. In the process_message method, add this code after classifying intent:

# Classify intent using the LLM-based classifier
intents = await self.intent_classifier.classify_intent(message)
top_intent = max(intents.items(), key=lambda x: x[1])[0]
top_score = max(intents.items(), key=lambda x: x[1])[1]

# Store intent and confidence in patient_data for debug info
patient_data['intent_classification'] = {
    'intent': top_intent,
    'confidence': top_score
}

logger.info(f"User message: {message}")


# 2. In the _generate_diagnostic_info method, update the debug_info template:

# Get intent classification information
intent_classification = patient_data.get('intent_classification', {'intent': intent, 'confidence': 0.0})
intent_name = intent_classification.get('intent', intent)
intent_confidence = intent_classification.get('confidence', 0.0)
intent_confidence_text = f"{intent_confidence:.2f} ({intent_confidence * 100:.1f}%)"

# Then use intent_name and intent_confidence_text in the debug_info template:
debug_info = f"""
------------------------------------------
DEBUG INFORMATION
------------------------------------------
Intent Classification: {intent_name} (confidence: {intent_confidence_text})
Current Dialogue State: {current_state}
OpenAI Model Used: {response_model}

Question Statistics:
• Total Questions Asked: {followup_count}
• Symptom-Related Questions: {symptom_related_questions}
• Answered Questions: {answered_questions}

Diagnosis Information:
• Primary Diagnosis: {diagnosis_name} (confidence: {confidence_text})
• Verification Trigger: {verification_trigger}

Differential Diagnoses:
{differential_text}
"""
