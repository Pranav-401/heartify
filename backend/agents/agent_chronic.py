import os
import requests
import json
import re

def predict_chronic_disease(heart_rate, stress, emotions):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return json.dumps({"error": "Groq API key missing – add to .env"})

    dominant = max(emotions, key=emotions.get) if emotions else "neutral"

    prompt = f"""Patient:
Heart Rate: {heart_rate} bpm
Stress Level: {round(stress * 100)}%
Dominant Emotion: {dominant}

First, estimate overall chronic disease risk probability (0.0 to 1.0).
Then, list exactly 4 most likely chronic disease risks.
Format:
Overall Probability: 0.X

1. Disease name – short reason
2. Disease name – short reason
3. Disease name – short reason
4. Disease name – short reason

Be realistic and evidence-based.
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 300
            },
            timeout=20
        )

        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"].strip()
            # Extract probability
            prob_match = re.search(r"Overall Probability: (\d\.\d+)", text)
            probability = float(prob_match.group(1)) if prob_match else 0.5

            # Extract diseases
            lines = [line.strip() for line in text.split("\n") if line.strip() and line[0].isdigit()]

            return json.dumps({
                "overall_probability": probability,
                "diseases": lines[:4]
            })
        
        error_msg = response.json().get("error", {}).get("message", "Unknown API error") if response.status_code != 200 else "Empty response"
        return json.dumps({"error": f"Groq API error: {error_msg}"})
    except Exception as e:
        return json.dumps({"error": "Network error – check internet or API key"})