import tensorflow as tf
from matplotlib import pyplot as plt
from keras.layers import Input

def unet_model(input_shape):
    inputs = Input(input_shape)

    # Contracting Path
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    drop4 = tf.keras.layers.Dropout(0.3)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    drop5 = tf.keras.layers.Dropout(0.3)(conv5)

    # # Expansive Path
    up6 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(drop5)
    up6 = tf.keras.layers.concatenate([up6, drop4], axis=3)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv7)
    up8 = tf.keras.layers.concatenate([up8, conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv8)
    up9 = tf.keras.layers.concatenate([up9, conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = tf.keras.layers.Conv2D(23, (1, 1), padding='same')(conv9)

    # outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv9])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

# Define input shape
input_shape = (96, 128, 3)

# Create the model
model = unet_model(input_shape)

# Display the model summary
# model.summary()

model.load_weights('haharoad.hdf5')


import imageio.v2 as im


img1 = im.imread('1.png')
# mask = im.imread('CameraMask/000026.png')
img = tf.io.read_file('1.png')
img = tf.image.decode_png(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = img[tf.newaxis, ...]

resized_image = tf.image.resize(img, (96, 128), method='nearest')

print(resized_image.shape)
pred= model.predict(resized_image)
pred_mask = tf.argmax(pred, axis=-1)
pred_mask = pred_mask[..., tf.newaxis]


fig, arr = plt.subplots(1, 2, figsize=(14, 10))
arr[0].imshow(img1)
arr[0].set_title('Image')
# arr[1].imshow(mask[:, :, 0])
# arr[1].set_title('Segmentation')
arr[1].imshow(pred_mask[0])
arr[1].set_title('predict')

plt.show()


