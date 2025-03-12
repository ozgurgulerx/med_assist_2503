import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Generate a user ID
response = requests.post(f"{BASE_URL}/generate_user_id")
user_id = response.json()["user_id"]
print(f"Generated user ID: {user_id}")

# Function to send a message and get response
def send_message(message):
    print(f"\nSending: '{message}'")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"user_id": user_id, "message": message}
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    response_data = response.json()
    bot_response = response_data.get("response", "")
    debug_info = response_data.get("debug_info", "")
    
    print(f"Bot: '{bot_response}'")
    if debug_info:
        print(f"\nDEBUG INFO:\n{debug_info}")
    
    return bot_response, debug_info

# Headache conversation flow
conversation = [
    "I have a headache",  # Initial symptom
    "Yes, I have sensitivity to light and some nausea",  # Migraine indicators
    "It's throbbing and on one side of my head",  # More migraine indicators
    "It started about 3 hours ago",  # Timing
    "About 7 out of 10",  # Pain level
    "Yes, I've had similar headaches before",  # History
    "No, I haven't taken any medication yet",  # Treatment
    "I sometimes see flashing lights before it starts",  # Aura
    "About once a month",  # Frequency
    "Yes, stress seems to trigger it",  # Triggers
    "No other medical conditions"  # Medical history
]

# Run the conversation
print("Starting headache diagnosis test...\n")

for message in conversation:
    response, debug = send_message(message)
    
    # Check if we've reached a diagnosis
    if debug and "Primary Diagnosis:" in debug:
        if "None" not in debug.split("Primary Diagnosis:")[1].split("\n")[0]:
            print("\n✅ DIAGNOSIS REACHED!")
            diagnosis_line = debug.split("Primary Diagnosis:")[1].split("\n")[0].strip()
            print(f"Diagnosis: {diagnosis_line}")
            break
    
    # Add a small delay between messages
    time.sleep(2)

print("\nHeadache test completed.")
