import os
import requests
import json

def suggest_diet_plan(heart_rate, stress, emotions, chronic_prediction):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return json.dumps({"error": "Groq API key missing"})

    try:
        chronic = json.loads(chronic_prediction)
        if "error" in chronic:
            return json.dumps({"error": chronic["error"]})
        diseases = "\n".join(chronic.get("diseases", []))
    except:
        diseases = "General health concerns"

    prompt = f"""Top risks:
{diseases}

Patient: HR {heart_rate} bpm, Stress {round(stress*100)}%

Suggest a simple daily diet plan to reduce these risks.
Include Breakfast, Lunch, Dinner, Snacks.
Short and practical.
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 300
            },
            timeout=20
        )

        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"].strip()
            if text:
                return json.dumps({"diet": text})

        error_msg = response.json().get("error", {}).get("message", "Unknown error")
        return json.dumps({"error": f"Groq API error: {error_msg}"})
    except Exception as e:
        return json.dumps({"error": "Network error"})