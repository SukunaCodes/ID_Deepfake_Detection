import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

print('TensorFlow version: ', tf.__version__)

dataset_path = '.\\split_dataset\\'
tmp_debug_path = '.\\tmp_debug\\'
print('Creating Directory... ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)


def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only


from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from efficientnet.tfkeras import EfficientNetB0
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size_num,
    shuffle=True
)

val_datagen = ImageDataGenerator(
    rescale=1 / 255
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size_num,
    shuffle=True,
)

test_datagen = ImageDataGenerator(
    rescale=1 / 255
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=['real', 'fake'],
    target_size=(input_size, input_size),
    color_mode='rgb',
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# Train the CNN classifier
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_filepath = '.\\tmp_checkpoint\\'
print('Creating Directory... ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath, 'IS_deepfake_detect_model.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]

# Train the neural network
num_epochs = 20
history = model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)
print(history.history)

# Here we plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


# Load the saved IS deepfake detect model.
IS_project_model = load_model(os.path.join(checkpoint_filepath, 'IS_deepfake_detect_model.h5'))

# Begin generating predictions
test_generator.reset()
preds = IS_project_model.predict(
    test_generator,
    verbose=1
)

test_results = pd.DataFrame({
    'Filename': test_generator.filenames,
    'Prediction': preds.flatten()
})
print(test_results)