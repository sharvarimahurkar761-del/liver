# data.py
import pandas as pd
import numpy as np

def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'Age': np.random.randint(20, 80, 200),
        'Bilirubin': np.random.uniform(0.3, 3.0, 200),
        'Alkaline_Phosphotase': np.random.randint(150, 400, 200),
        'SGPT_Alanine_Aminotransferase': np.random.randint(10, 100, 200),
        'Albumin': np.random.uniform(2.5, 5.0, 200),
        'Albumin_and_Globulin_Ratio': np.random.uniform(0.8, 2.5, 200),
        'Total_Proteins': np.random.uniform(5.0, 8.0, 200),
        'Age_Bilirubin_Ratio': np.random.uniform(5, 25, 200),  # example derived feature
        'Target': np.random.randint(0, 2, 200)
    })
    X = data.drop('Target', axis=1)
    y = data['Target']
    return X, y
