# Replace lines 490-493 in core_bot.py with this code

# Get intent classification information
intent_classification = patient_data.get('intent_classification', {'intent': intent, 'confidence': 0.0})
intent_name = intent_classification.get('intent', intent)
intent_confidence = intent_classification.get('confidence', 0.0)
intent_confidence_text = f"{intent_confidence:.2f} ({intent_confidence * 100:.1f}%)"

# Format the debug information section
