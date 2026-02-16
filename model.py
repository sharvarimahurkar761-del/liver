# model.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf_params = {'n_estimators':[100,200], 'max_depth':[None,5,10], 'min_samples_split':[2,5]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    joblib.dump(rf_model, "liver_rf_model.pkl")

    # Gradient Boosting
    gb_params = {'n_estimators':[100,200], 'learning_rate':[0.05,0.1], 'max_depth':[3,5]}
    gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=3, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    gb_model = gb_grid.best_estimator_
    joblib.dump(gb_model, "liver_gb_model.pkl")

    return rf_model, gb_model, X_train, X_test, y_train, y_test, rf_grid.best_params_, gb_grid.best_params_
