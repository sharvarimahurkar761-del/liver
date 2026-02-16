# data.py
import pandas as pd
import numpy as np

def generate_liver_dataset(n_samples=583, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 80, n_samples),
        'Total_Bilirubin': np.random.uniform(0.3, 6.0, n_samples),
        'Direct_Bilirubin': np.random.uniform(0.1, 3.0, n_samples),
        'Alkaline_Phosphotase': np.random.randint(100, 400, n_samples),
        'SGPT_Alanine_Aminotransferase': np.random.randint(10, 100, n_samples),
        'SGOT_Aspartate_Aminotransferase': np.random.randint(10, 120, n_samples),
        'Total_Proteins': np.random.uniform(4.5, 8.5, n_samples),
        'Albumin': np.random.uniform(2.0, 5.5, n_samples),
        'Albumin_and_Globulin_Ratio': np.random.uniform(0.8, 2.5, n_samples),
        'Gender': np.random.choice([0,1], n_samples),  # 0 = Female, 1 = Male
    })
    # Target probabilities for realism
    prob_disease = (
        0.3*(df['Total_Bilirubin']>1.2).astype(int) +
        0.2*(df['Direct_Bilirubin']>0.5).astype(int) +
        0.2*(df['SGPT_Alanine_Aminotransferase']>50).astype(int) +
        0.3*(df['SGOT_Aspartate_Aminotransferase']>50).astype(int)
    )
    df['target'] = (np.random.rand(n_samples) < prob_disease).astype(int)
    return df
