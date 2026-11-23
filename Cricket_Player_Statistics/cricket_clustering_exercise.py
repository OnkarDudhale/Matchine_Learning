import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib 


base_url = Path(__file__).resolve().parent
def phase_1_discovery_and_setup():
    try:
        df = pd.read_csv(base_url/'cricket_player_stats.csv')
        
        categorical_features = df['player_name']
        df = df.drop('player_id', axis=1)
        numerical_features = df.select_dtypes('number')
        print(numerical_features)
        
        return numerical_features, df
        
    except FileNotFoundError:
        print('Error: File Not found')
        exit(0)
    except Exception as e:
        print('Error: unexpected error')        
        raise

def phase_2_feature_scaling(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler
    
def phase_3_kmeans_model_induction(X_scaled, df_original):
    
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X_scaled) 
    
    df_clustered = df_original.copy()
    
    df_clustered['cluster_id'] = labels
    print(df_clustered)    
    
    return model, labels, df_clustered 

def phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered):
    score = silhouette_score(X_scaled, labels)
    print(score)
    
    centroid = df_clustered.groupby('cluster_id').mean(numeric_only = True)
    print(centroid)
    
def phase_5_joblib_model_persistence(model, scaler, feature_columns):
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_columns, 'features.pkl')
    
    loaded_model = joblib.load('model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_features = joblib.load('features.pkl')
    
    sample_inference = pd.DataFrame(
        [[135, 450, 14.2, 240,5.3 ]], columns=loaded_features
    )
    
    scaled_data = loaded_scaler.transform(sample_inference)
    predictions = loaded_model.predict(scaled_data)
    print(predictions) 
        
def run_cricket_clustering_pipeline():
    """
    Facade function coordinating the cricket player segmentation pipeline.
    """
    print("=" * 65)
    print("STARTING CRICKET PLAYER ROLE CLUSTERING LIFECYCLE (K-Means)")
    print("=" * 65)
    
    # Executing phases chronologically
    X, df_original = phase_1_discovery_and_setup()
    X_scaled, scaler = phase_2_feature_scaling(X)
    model, labels, df_clustered = phase_3_kmeans_model_induction(X_scaled, df_original)
    phase_4_cluster_evaluation_and_profiling(X_scaled, labels, df_clustered)
    phase_5_joblib_model_persistence(model, scaler, list(X.columns))


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_cricket_clustering_pipeline()