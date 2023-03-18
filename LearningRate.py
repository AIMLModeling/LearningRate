# Acquire MNIST data
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Reshape data
train_images = np.reshape(train_images, (-1, 784))
test_images = np.reshape(test_images, (-1, 784))
# Normalize data
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
# Autoencoder model
input_dim = 28*28
latent_vec_dim = 16
input_layer = Input(shape=(input_dim,))
# Define the autoencoder architecture
# First build the encoder
enc_layer_1 = Dense(latent_vec_dim, activation='tanh')(input_layer)
encoder = enc_layer_1
# Then build the decoder
dec_layer_1 = Dense(input_dim, activation='sigmoid')(encoder)
decoder = dec_layer_1
# Connect both encoder and decoder
autoencoder = Model(input_layer, decoder, name="AE_latent_dim_16")
# Latent representation (Optional)
latent_model = Model(input_layer, encoder)
# Compile the autoencoder model
learningRate=0.001
caseNo=2
if caseNo == 1:
    learningRate=0.0005
elif caseNo == 2:
    learningRate=0.001
elif caseNo == 3:
    learningRate=0.01
elif caseNo == 4:
    learningRate=0.02
elif caseNo == 5:
    learningRate=0.9
elif caseNo == 6:
    learningRate=0.0017
autoencoder.compile(optimizer=Adam(learning_rate=learningRate),
                    loss='binary_crossentropy')
# Train the autoencoder with MNIST data
history = autoencoder.fit(train_images, train_images, 
                          epochs=70, batch_size=128,
                          shuffle=True, 
                          validation_data=(test_images, test_images))
# Plot training and validation loss scores
# against the number of epochs.
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.ylabel('Binary Cross Entropy Loss')
plt.xlabel("Epoch (learning rate=" + str(learningRate) + ")")
plt.title('Autoencoder Reconstruction Loss', pad=13)
plt.legend(loc='upper right')