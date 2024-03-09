from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
import random
from matplotlib import pyplot as plt
import glob

# u-net model
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model



# print(os.listdir("/content/drive/MyDrive/unet/new/"))

SIZE = 256  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = []

for directory_path in glob.glob("/content/drive/MyDrive/unet/new/train/*"):
      if directory_path.endswith(".jpg"):
          # print(directory_path)
          img_path=directory_path

          img = cv2.imread(img_path, cv2.IMREAD_COLOR)
          img = cv2.resize(img, (SIZE, SIZE))
          img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          train_images.append(img)
          # print(img_path)

          label_path=img_path.replace(".jpg", "_mask.png")
          # print(label_path)
          label = cv2.imread(label_path)
          label = cv2.resize(label, (SIZE, SIZE))
          label=cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
          train_labels.append(label)

# plt.imshow(train_images[10])
# plt.imshow(train_labels[1]*255)
#Convert lists to arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)



valid_images = []
valid_labels = []
for directory_path in glob.glob("/content/drive/MyDrive/unet/new/valid/*"):
    if directory_path.endswith(".jpg"):
          # print(directory_path)
          img_path=directory_path

          img = cv2.imread(img_path, cv2.IMREAD_COLOR)
          img = cv2.resize(img, (SIZE, SIZE))
          img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          valid_images.append(img)
          # print(img_path)

          label_path=img_path.replace(".jpg", "_mask.png")
          # print(label_path)
          label = cv2.imread(label_path)
          label = cv2.resize(label, (SIZE, SIZE))
          label=cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
          valid_labels.append(label)

# plt.imshow(test_images[3])
# plt.imshow(test_labels[3]*255)

#Convert lists to arrays
valid_images = np.array(valid_images)
valid_labels = np.array(valid_labels)

X_train, Y_train, X_test, Y_test = train_images, train_labels, valid_images, valid_labels

# Normalize pixel values to between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

#Sanity check, view few mages
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE,3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(Y_train[image_number]*255, (SIZE, SIZE,1)), cmap='gray')
plt.show()

IMG_HEIGHT = SIZE
IMG_WIDTH  = SIZE
IMG_CHANNELS = 3

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


#If starting with pre-trained weights.
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')

my_callbacks = [
    keras.callbacks.EarlyStopping(patience=3),
    keras.callbacks.ModelCheckpoint(filepath='/content/drive/MyDrive/unet/new/model.{epoch:02d}-{val_loss:.2f}.h5',save_best_only=True,),
    keras.callbacks.TensorBoard(log_dir='./logs'),
]

history = model.fit(X_train, Y_train,
                    batch_size = 16,
                    verbose=1,
                    epochs=50,
                    validation_data=(X_test, Y_test),
                    shuffle=False,
                    callbacks=my_callbacks)

model.save('road.hdf5')

#Evaluate the model


	# evaluate model
_, acc = model.evaluate(X_test, Y_test)
print("Accuracy = ", (acc * 100.0), "%")


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# acc = history.history['acc']
acc = history.history['accuracy']
# val_acc = history.history['val_acc']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

y_pred_thresholded = np.squeeze(y_pred_thresholded, axis=-1)

# Now perform the logical operations
intersection = np.logical_and(Y_test, y_pred_thresholded)
union = np.logical_or(Y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)

print("Intersection over Union (IoU) score:", iou_score)

test_images_new = []
test_labels_new = []
for directory_path in glob.glob("/content/drive/MyDrive/unet/new/test/*"):
    if directory_path.endswith(".jpg"):
          # print(directory_path)
          img_path=directory_path

          img = cv2.imread(img_path, cv2.IMREAD_COLOR)
          img = cv2.resize(img, (SIZE, SIZE))
          img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          test_images_new.append(img)
          # print(img_path)

          label_path=img_path.replace(".jpg", "_mask.png")
          # print(label_path)
          label = cv2.imread(label_path)
          label = cv2.resize(label, (SIZE, SIZE))
          label=cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
          test_labels_new.append(label)


#Predict on a few images
model = get_model()
model.load_weights('road.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

test_img_number1 = random.randint(0, len(test_images_new)-1)
test_img1 = test_images_new[test_img_number1]
ground_truth=test_labels_new[test_img_number1]
test_img_norm=test_img1[:,:,:][:,:,:,None]

test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)


plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.title('Testing Image')
plt.imshow(test_img1[:,:,:], cmap='gray')
plt.subplot(132)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:], cmap='gray')
plt.subplot(133)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()