import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from sklearn.preprocessing import StandardScaler

# Define the VAE model
def build_vae(input_dim, latent_dim):
    inputs = Input(shape=(input_dim,))
    h = Dense(64, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    decoder_h = Dense(64, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(inputs, x_decoded_mean)
    
    # Define the VAE loss function
    def vae_loss(inputs, x_decoded_mean, z_mean, z_log_var):
        xent_loss = binary_crossentropy(inputs, x_decoded_mean)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return tf.reduce_mean(xent_loss + kl_loss)

    # Create a custom layer to compute and add the loss
    class VAELossLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(VAELossLayer, self).__init__(**kwargs)
        
        def call(self, inputs):
            x_decoded_mean, z_mean, z_log_var = inputs
            loss = vae_loss(inputs[0], x_decoded_mean, z_mean, z_log_var)
            self.add_loss(loss)
            return x_decoded_mean

    vae_outputs = VAELossLayer()([x_decoded_mean, z_mean, z_log_var])
    vae = Model(inputs, vae_outputs)
    
    return vae, inputs, x_decoded_mean, z_mean, z_log_var

# Custom training function
def train_vae(vae, X_train, epochs=50, batch_size=32):
    optimizer = tf.keras.optimizers.Adam()
    
    @tf.function
    def train_step(batch_x):
        with tf.GradientTape() as tape:
            x_decoded_mean = vae(batch_x, training=True)
            loss = vae.losses[0]  # Access the custom loss function added to the model
        grads = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae.trainable_variables))
        return loss

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            loss = train_step(batch_x)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.numpy()}')

# Load and preprocess data
data = pd.read_csv('fintech_data1.csv')
features = data.drop(columns=['fraud_bool'])
features = pd.get_dummies(features)  # Convert categorical variables to dummy/indicator variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Define and compile the VAE model
input_dim = X_scaled.shape[1]
latent_dim = 10
vae, inputs, x_decoded_mean, z_mean, z_log_var = build_vae(input_dim, latent_dim)

# Compile the model with an optimizer
vae.compile(optimizer='adam')

# Train the model
train_vae(vae, X_scaled, epochs=50, batch_size=32)

# Save the model
vae.save('vae_model.h5')
