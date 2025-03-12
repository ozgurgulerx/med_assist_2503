# Update these sections in core_bot.py to fix the diagnosis display issue

# 1. Update the diagnosis information retrieval (around line 408)
# Replace these lines:
# diagnosis_info = patient_data.get("diagnosis", {})
# diagnosis_name = diagnosis_info.get("name", "Pending diagnosis")
# confidence = diagnosis_info.get("confidence", 0.0)
# confidence_text = f"{confidence:.2f} ({confidence * 100:.1f}%)"

# With these lines:
diagnosis_info = patient_data.get("diagnosis", {})
diagnosis_name = diagnosis_info.get("name", "Pending diagnosis")
# Ensure diagnosis_name is never None in the display
if diagnosis_name is None:
    diagnosis_name = "Pending diagnosis"
confidence = diagnosis_info.get("confidence", 0.0)
confidence_text = f"{confidence:.2f} ({confidence * 100:.1f}%)"

# 2. Update the differential diagnoses formatting (around line 416)
# Replace these lines:
# if differential_diagnoses:
#     diff_items = []
#     for diag in differential_diagnoses:
#         if isinstance(diag, dict):
#             name = diag.get("name", "Unknown")
#             conf = diag.get("confidence", 0.0)
#             diff_items.append(f"• {name}: {conf:.2f} ({conf * 100:.1f}%)")
#         else:
#             diff_items.append(f"• {diag}")
#     differential_text = "\n".join(diff_items)
# else:
#     differential_text = "No alternative diagnoses identified"

# With these lines:
if differential_diagnoses:
    diff_items = []
    for i, diag in enumerate(differential_diagnoses):
        if isinstance(diag, dict):
            name = diag.get("name", "Unknown")
            # Ensure name is never None in the display
            if name is None:
                name = "Unknown condition"
            conf = diag.get("confidence", 0.0)
            rank = "Secondary" if i == 0 else "Tertiary" if i == 1 else f"Alternative #{i+1}"
            diff_items.append(f"• {rank} Diagnosis: {name} (confidence: {conf:.2f} ({conf * 100:.1f}%))")
        else:
            diff_items.append(f"• Alternative: {diag}")
    differential_text = "\n".join(diff_items)
else:
    differential_text = "No alternative diagnoses identified"

# 3. Update the debug information section (around line 509)
# Replace these lines:
# Diagnosis Information:
# • Primary Diagnosis: {diagnosis_name} (confidence: {confidence_text})
# • Verification Trigger: {verification_trigger}
# Check if this is a greeting that should trigger a session reset
if top_intent == "greeting" and self.dialog_manager.get_user_state(user_id) in ["farewell", "generating_diagnosis"]:
    logger.info(f"Detected greeting after conversation completion, resetting session for user {user_id}")
    self.reset_session(user_id)
    # Since we've reset the session, update our local references
    history = self.get_chat_history(user_id)
    user_data = self.get_user_data(user_id)
    patient_data = initialize_patient_data_if_needed(user_data)

# Special handling for greeting and smallTalk intents - don't process as symptoms
if top_intent in ["greeting", "smallTalk"]:
    logger.info(f"{top_intent} detected, skipping symptom processing")
    # ... rest of the code ...
# With these lines:
Diagnosis Information:
• Primary Diagnosis: {diagnosis_name} (confidence: {confidence_text})
• Verification Trigger: {verification_trigger}

Differential Diagnoses:
{differential_text}
