import pandas as pd
import numpy as np

np.random.seed(42)
num_samples = 1000

data = {
    "Age": np.random.randint(18, 80, num_samples),
    "Gender": np.random.choice(["Male", "Female"], num_samples),
    "BMI": np.round(np.random.uniform(18.5, 35, num_samples), 1),
    "Symptom_1": np.random.choice(
        ["Fever", "Cough", "Chest Pain", "Fatigue", "Headache", "None"], num_samples
    ),
    "Symptom_2": np.random.choice(
        ["Nausea", "Shortness of Breath", "Dizziness", "Sneezing", "Swelling", "None"],
        num_samples,
    ),
    "Blood_Pressure": np.random.choice(
        ["120/80", "130/85", "140/90", "150/95", "160/100"], num_samples
    ),
    "Health_Condition": np.random.choice(
        ["Healthy", "Mild Condition", "Chronic Illness", "Emergency"], num_samples,
        p=[0.5, 0.3, 0.15, 0.05]  # Probabilities for realistic distribution
    ),
}

df = pd.DataFrame(data)

df.to_csv("patient_health_dataset.csv", index=False)
print("Dataset created and saved as 'patient_health_dataset.csv'.")