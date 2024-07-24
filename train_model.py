import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('fintech_data1.csv')  # Replace with your actual file path

# Inspect the data
print("Columns in the dataset:", data.columns)

# Assume 'fraud_bool' is the target variable and the rest are features
target = data['fraud_bool']
features = data.drop(columns=['fraud_bool'])

# Identify categorical columns and numeric columns
categorical_columns = features.select_dtypes(include=['object']).columns
numeric_columns = features.select_dtypes(include=['number']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # Handle unknown categories
    ])

# Preprocess the features
X = preprocessor.fit_transform(features)
y = target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype('int32')
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the model
model.save('model.h5')

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    joblib.dump(preprocessor, f)
