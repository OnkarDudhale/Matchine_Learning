import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

base_url = Path(__file__).resolve().parent
def phase_1_discovery_and_target_identification():
    try:
        
        df = pd.read_csv(base_url/'housing_ml_data.csv')
        y = df['price_usd']
        X  = df.drop(['house_id', 'price_usd'], axis=1)
        
        print(y.shape)
        print(X.shape)
        
        return X, y
        
    except FileNotFoundError:
        print('Error: File not found')
    except Exception as e:
        print('Error: unexpected error', e) 
        raise
    
def phase_2_feature_engineering_and_encoding(X):
    categorical_columns = X.select_dtypes(['object', 'string']).columns    
    ord = OrdinalEncoder()
    X[categorical_columns] = ord.fit_transform(X[categorical_columns])
    
    X.loc[X['num_bedrooms'] == 0, 'num_bedrooms'] = 1
    
    X['sqft_per_room'] = X['square_feet'] / X['num_bedrooms']
    print(X.head())
    
    return X

def phase_3_scaling_and_train_test_split(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
    scaler = StandardScaler()
    
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    return X_tr_scaled, X_te_scaled, y_train, y_test, scaler 
    
def phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te):
    model = RandomForestRegressor()
    model.fit(X_tr_scaled, y_tr)
    
    predictions = model.predict(X_te_scaled)
    print(predictions)
    print('MSE:', mean_squared_error(y_te, predictions))
    print('MAE:', mean_absolute_error(y_te, predictions))
    print('R2 Score:', r2_score(y_te, predictions))    
    
    
    return model

def phase_5_joblib_model_persistence(model, scaler, X_encoded_columns):
    
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X_encoded_columns, 'X_encoded_columns.pkl')
    
    loaded_model = joblib.load('model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_encoded_columns = joblib.load('X_encoded_columns.pkl')
    
    sample_inference = pd.DataFrame(
        [[ 1.0, 2100, 3, 15, 1.0, 700.000000]], columns=X_encoded_columns
    )    
    
    scaled_data = loaded_scaler.fit_transform(sample_inference)
    predictions = loaded_model.predict(scaled_data)
    print(predictions)
    
def run_ml_pipeline():
    """
    Facade function coordinating the entire machine learning pipeline.
    """
    print("=" * 65)
    print("STARTING END-TO-END MACHINE LEARNING LIFECYCLE ")
    print("=" * 65)

    # Executing phases chronologically
    X, y = phase_1_discovery_and_target_identification()
    X_encoded = phase_2_feature_engineering_and_encoding(X)
    X_tr_scaled, X_te_scaled, y_tr, y_te, scaler = phase_3_scaling_and_train_test_split(X_encoded, y)
    model = phase_4_model_training_and_testing(X_tr_scaled, X_te_scaled, y_tr, y_te)
    phase_5_joblib_model_persistence(model, scaler, X_encoded.columns.tolist())

if __name__ == '__main__':
    run_ml_pipeline()