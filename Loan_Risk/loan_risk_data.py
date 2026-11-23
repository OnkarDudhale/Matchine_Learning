import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import joblib
import numpy as np

base_URL = Path(__file__).resolve().parent

def phase_1_discovery_and_target_identification():
    try:
        df = pd.read_csv(base_URL/'loan_risk.csv')
        
        if df.empty:
            print('File is empty..')
            return
            
        print(df.info())    
        df = df.drop('applicant_id', axis=1)
        
        X = df[['annual_income_k', 'credit_score', 'employment_years', 'loan_amount_k', 'housing_status']]    
        y = df['default_flag']
        
        print(df['default_flag'].value_counts())
        
        # normalize gives proportion
        print(df['default_flag'].value_counts(normalize=True) * 100)
        
        print(df)
        
        return X, y
    
    except FileNotFoundError:
        print('Error: File not found')
    except Exception as e:
        print(f'Unexpected error: {e}')
        raise
        
def phase_2_feature_engineering_and_encoding(X):
    
    # one hot encoding
    X_encoding = pd.get_dummies(X['housing_status'], sparse=True, drop_first=True, dtype=int)   
    
    # concating features and housing_status encoded values
    X = pd.concat([ 
            X.drop(columns=['housing_status']), X_encoding,
        ], axis=1)
    
    X['loan_to_income_ratio'] = X['loan_amount_k'] / X['annual_income_k']   
    
    print(X.head())
    return X
    
def phase_3_scaling_and_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # does not substract with mean because to preserve sparse values
    scaler = StandardScaler(with_mean=False)
    
    # scaling data on std(how many std a value from mean)
    # StandardScaler to bring numerical features onto a similar scale
    X_train_Scaled = scaler.fit_transform(X_train)
    X_test_Scaled = scaler.transform(X_test)
    
    print(np.round(X_train_Scaled, 2))
    print(np.round(X_test_Scaled, 2))
    return scaler, X_train_Scaled, X_test_Scaled, y_train, y_test
    
def phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te):
    model = LogisticRegression()
    model.fit(X_tr_scaled, y_tr)
    
    predictions = model.predict(X_te_scaled)
    print(predictions)
    print(y_te.tolist())
    accuracy_sc = accuracy_score(y_te, predictions)
    print(accuracy_sc)
    
    class_report = classification_report(y_te, predictions)
    print(class_report)
    
    return model

def phase_5_joblib_model_persistence(model, scaler, feature_names, X_te_scaled):
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("Model saved successfully!")
    print("Scaler saved successfully!")
    print("Feature names saved successfully!")
    
    loaded_model = joblib.load('logistic_regression_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_features = joblib.load('feature_names.pkl')
    
    print("Model and scaler loaded successfully!")
    
    # print(X_te_scaled)
    sample_data = X_te_scaled[0:1]
    print(sample_data)
    
    prediction = loaded_model.predict(sample_data)
    print("Sample prediction:", prediction)
    
    return loaded_model    
    
def run_ml_pipeline():
    """
    Facade function coordinating the entire loan risk classification pipeline.
    """
    print("=" * 65)
    print("STARTING LOAN RISK CLASSIFICATION LIFECYCLE (Logistic Regression)")
    print("=" * 65)
    
    # Executing phases chronologically
    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    scaler, X_tr_scaled, X_te_scaled, y_tr, y_te = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist(), X_te_scaled)


if __name__ == '__main__':
    run_ml_pipeline()
    
    
'''
One-Hot_Encoding

Parameter
sparse=True
drop_first=True
dtype=int

Purpose
Saves memory for sparse encoded data
Removes one category to avoid redundant features
Produces 1 and 0 instead of True and False

'''
        