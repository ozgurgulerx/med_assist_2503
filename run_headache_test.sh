#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Generate a user ID
echo "Generating user ID..."
USER_ID=$(curl -s -X POST http://localhost:8000/generate_user_id | jq -r '.user_id')
echo "User ID: $USER_ID"

# Function to send a message and get response
send_message() {
    local message="$1"
    echo -e "\nSending: '$message'"
    response=$(curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d "{\"user_id\": \"$USER_ID\", \"message\": \"$message\"}")
    
    bot_response=$(echo "$response" | jq -r '.response')
    debug_info=$(echo "$response" | jq -r '.debug_info')
    
    echo "Bot: '$bot_response'"
    if [ "$debug_info" != "" ]; then
        echo -e "\nDEBUG INFO:\n$debug_info"
    fi
    
    # Check if we've reached a diagnosis
    if [[ "$debug_info" == *"Primary Diagnosis:"* ]]; then
        diagnosis=$(echo "$debug_info" | grep -A 1 "Primary Diagnosis:" | tail -n 1)
        if [[ "$diagnosis" != *"None"* ]]; then
            echo -e "\n✅ DIAGNOSIS REACHED!"
            echo "Diagnosis: $diagnosis"
            return 1
        fi
    fi
    
    return 0
}

# Headache conversation flow
echo "Starting headache diagnosis test...\n"

# Initial symptom
send_message "I have a headache" || exit 0
sleep 2

# Migraine indicators
send_message "Yes, I see flashing lights in my vision before the headache starts" || exit 0
sleep 2

# More migraine indicators
send_message "It's throbbing and on one side of my head" || exit 0
sleep 2

# Timing
send_message "It started about 3 hours ago" || exit 0
sleep 2

# Pain level
send_message "About 7 out of 10" || exit 0
sleep 2

# History
send_message "Yes, I've had similar headaches before" || exit 0
sleep 2

# Treatment
send_message "No, I haven't taken any medication yet" || exit 0
sleep 2

# Frequency
send_message "About once a month" || exit 0
sleep 2

# Triggers
send_message "Yes, stress seems to trigger it" || exit 0
sleep 2

# Medical history
send_message "No other medical conditions" || exit 0
sleep 2

echo -e "\nHeadache test completed."
