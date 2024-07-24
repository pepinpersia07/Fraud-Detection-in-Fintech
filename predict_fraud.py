import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

# Define and register the custom sampling function
@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae_model(input_shape, latent_dim=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    
    decoder_inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(decoder_inputs)
    outputs = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
    
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    
    return vae

def load_model():
    vae = tf.keras.models.load_model('vae_model.h5', custom_objects={'sampling': sampling})
    print("Model loaded successfully.")
    return vae

def predict_fraud():
    vae = load_model()
    
    # Load data from CSV
    df = pd.read_csv('new_transaction.csv')
    
    # Print the first few rows of the DataFrame and data types
    print("Loaded DataFrame:")
    print(df.head())
    print("Data types before conversion:")
    print(df.dtypes)
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns
    
    # Convert numeric columns to numeric types
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    print("DataFrame after conversion to numeric:")
    print(df_numeric.head())
    
    # Handle categorical columns
    df_categorical = df[categorical_cols]
    df_encoded = pd.get_dummies(df_categorical)
    print("Encoded categorical DataFrame:")
    print(df_encoded.head())
    
    # Combine numeric and encoded categorical columns
    df_combined = pd.concat([df_numeric, df_encoded], axis=1)
    
    # Drop rows with NaNs
    df_combined_cleaned = df_combined.dropna()
    print("DataFrame after dropna:")
    print(df_combined_cleaned.head())
    
    # Check if there is any data available
    if df_combined_cleaned.empty:
        print("No valid data available for prediction.")
        return
    
    # Convert to NumPy array and ensure it is of type float32
    input_data = df_combined_cleaned.values.astype(np.float32)
    
    # Check for NaNs or infinities in the input data
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
        raise ValueError("Input data contains NaNs or infinities.")
    
    # Retrieve the expected input shape from the model
    input_shape = vae.input_shape[1:]
    
    # Ensure the input data shape matches the model's expected input shape
    if input_data.shape[1] != input_shape[0]:
        print(f"Adjusting input data shape: expected {input_shape[0]}, but got {input_data.shape[1]}.")
        if input_data.shape[1] < input_shape[0]:
            padding = np.zeros((input_data.shape[0], input_shape[0] - input_data.shape[1]))
            input_data = np.concatenate([input_data, padding], axis=1)
        elif input_data.shape[1] > input_shape[0]:
            input_data = input_data[:, :input_shape[0]]
    
    # Check the adjusted shape of input data
    print(f"Input data shape after adjustment: {input_data.shape}")
    
    # Ensure there is enough data for prediction
    if input_data.shape[0] == 0:
        print("No data available for prediction after adjustment.")
        return
    
    # Make predictions
    try:
        predictions = vae.predict(input_data, verbose=0)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Prepare results for JSON output
    results = []
    for i, prediction in enumerate(predictions):
        fraud_percentage = round(np.mean(prediction) * 100, 2)  # Example: mean of predictions for percentage
        is_fraud = fraud_percentage > 50  # Example threshold
        
        result_entry = {
            "entry_id": i,
            "fraudPercentage": fraud_percentage,
            "isFraud": bool(is_fraud)  # Convert to standard Python boolean
        }
        results.append(result_entry)
    
    # Save results to JSON
    with open('predictions.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Predictions saved to 'predictions.json'.")

if __name__ == "__main__":
    predict_fraud()
