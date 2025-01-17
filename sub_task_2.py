import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# taking MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0  # Normalize to [0, 1]
batch_size = 64


dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=1024).batch(batch_size)


z_dim = 100  #noise
learning_rate = 0.0002
epochs = 5000

# Generator 
class Generator(tf.keras.Model):
    def __init__(self, z_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense4 = tf.keras.layers.Dense(784, activation='sigmoid')  # Output is 28x28 image

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

# Discriminator 
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(0.2))
        self.dense2 = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.2))
        self.dense3 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.2))
        self.dense4 = tf.keras.layers.Dense(1)  # Single output to tell fake/real.

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

# Instantiate models
generator = Generator(z_dim)
discriminator = Discriminator()

# Optimizers
gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate)

# Loss functions
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

# Training step
@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.uniform([batch_size, z_dim], -1.0, 1.0)

    # Train Discriminator 
    with tf.GradientTape() as disc_tape:
        fake_images = generator(noise)
        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images)
        disc_loss = discriminator_loss(real_output, fake_output)

    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    # Clip discriminator weights 
    for var in discriminator.trainable_variables:
        var.assign(tf.clip_by_value(var, -0.01, 0.01))

    # Train Generator
    with tf.GradientTape() as gen_tape:
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        gen_loss = generator_loss(fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    return disc_loss, gen_loss

# Function to generate and display images
def generate_and_plot_images(epoch, test_noise):
    generated_images = generator(test_noise, training=False).numpy()
    generated_images = generated_images.reshape(-1, 28, 28)

    plt.figure(figsize=(5, 5))
    for i in range(25):  # Display 25 images
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Epoch: {epoch}")
    plt.show()

# Training loop
test_noise = tf.random.uniform([25, z_dim], -1.0, 1.0)  # Fixed noise for visualization

for epoch in range(epochs):
    for real_images in dataset:
        disc_loss, gen_loss = train_step(real_images)

   
