# Add this code after line 608 in core_bot.py (after classifying intent)

# Store intent and confidence in patient_data for debug info
patient_data['intent_classification'] = {
    'intent': top_intent,
    'confidence': top_score
}
