import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
def load_and_preprocess_image(image_path):
 img = image.load_img(image_path, color_mode='grayscale')
 img_array = image.img_to_array(img)
 img_array = np.expand_dims(img_array, axis=0)
 img_array = img_array / 255.0
 return img_array
def make_generator(input_shape, output_shape):
 model = keras.Sequential([
 layers.Dense(128 * (output_shape[0] // 4) * (output_shape[1] // 4),
input_shape=input_shape),
 layers.Reshape((output_shape[0] // 4, output_shape[1] // 4, 128)),
 layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
 layers.LeakyReLU(alpha=0.2),
 layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'),
 layers.LeakyReLU(alpha=0.2),
 layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
 ])
 return model
def make_discriminator(input_shape):
 model = keras.Sequential([
 layers.Conv2D(32, kernel_size=3, strides=2, padding='same',
input_shape=input_shape),
 layers.LeakyReLU(alpha=0.2),
 layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
 layers.LeakyReLU(alpha=0.2),
 layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
 layers.LeakyReLU(alpha=0.2),
 layers.Flatten(),
 layers.Dense(1, activation='sigmoid')
 ])
 return model
class GAN(keras.Model):
 def __init__(self, discriminator, generator, latent_dim):
 super(GAN, self).__init__()
 self.discriminator = discriminator
 self.generator = generator
 self.latent_dim = latent_dim
 def compile(self, d_optimizer, g_optimizer, loss_fn):
 super(GAN, self).compile()
 self.d_optimizer = d_optimizer
 self.g_optimizer = g_optimizer
 self.loss_fn = loss_fn
 def train_step(self, real_images):
 batch_size = tf.shape(real_images)[0]

 noise = tf.random.normal([batch_size, self.latent_dim])
 generated_images = self.generator(noise)

 with tf.GradientTape() as tape:
 real_output = self.discriminator(real_images, training=True)
 fake_output = self.discriminator(generated_images, training=True)
 d_loss = self.loss_fn(tf.ones_like(real_output), real_output) + \
 self.loss_fn(tf.zeros_like(fake_output), fake_output)

 grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
 self.d_optimizer.apply_gradients(zip(grads,
self.discriminator.trainable_variables))

 noise = tf.random.normal([batch_size, self.latent_dim])

 with tf.GradientTape() as tape:
 generated_images = self.generator(noise, training=True)
 fake_output = self.discriminator(generated_images, training=True)
 g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

 grads = tape.gradient(g_loss, self.generator.trainable_variables)
 self.g_optimizer.apply_gradients(zip(grads,
self.generator.trainable_variables))

 return {"d_loss": d_loss, "g_loss": g_loss}
# Load and preprocess the image
image_path = './test.jpg' # Replace with your image path
Xtrain = load_and_preprocess_image(image_path)
print("Input image shape:", Xtrain.shape)
# Set up the GAN
latent_dim = 100
generator = make_generator((latent_dim,), Xtrain.shape[1:])
discriminator = make_discriminator(Xtrain.shape[1:])
gan = GAN(discriminator, generator, latent_dim)
gan.compile(
 d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
 g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
 loss_fn=keras.losses.BinaryCrossentropy()
)
# Train the GAN
batch_size = 1
dataset = tf.data.Dataset.from_tensor_slices(Xtrain).batch(batch_size)
epochs = 1000
for epoch in range(epochs):
 print(f"\nEpoch {epoch+1}")
 for batch in tqdm(dataset):
 gan.train_step(batch)
 if (epoch + 1) % 100 == 0:
 noise = tf.random.normal([1, latent_dim])
 generated_image = gan.generator(noise, training=False)
 plt.figure(figsize=(10, 10))
 plt.imshow(generated_image[0, :, :, 0], cmap='gray')
 plt.axis('off')
 plt.savefig(f'generated_image_epoch_{epoch+1}.png')
 plt.close()
# Display the original and generated images
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(Xtrain[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Generated Image')
noise = tf.random.normal([1, latent_dim])
generated_image = gan.generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
