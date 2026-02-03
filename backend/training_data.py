import pandas as pd
from datetime import datetime

# Create the data folder if it doesn't exist
import os
os.makedirs("data", exist_ok=True)

# Define columns for training data
columns = [
    "timestamp",
    "heart_rate",
    "stress",
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
    "ppg_mean_signal", "ppg_variance", "ppg_buffer_length",
    "chronic_risk_probability"
]

# Create empty DataFrame
df = pd.DataFrame(columns=columns)

# Save to Excel
file_path = "data/training_data.xlsx"
df.to_excel(file_path, index=False)

print(f"Excel file recreated: {file_path}")
print("Columns:", list(df.columns))