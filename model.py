# model.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
from data import load_data

def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)

    # Save models
    joblib.dump(rf, 'rf_model.pkl')
    joblib.dump(gb, 'gb_model.pkl')

    return rf, gb
