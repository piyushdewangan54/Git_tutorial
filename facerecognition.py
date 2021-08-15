# Used To initialized Neural Network
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Model Creation
classifier = Sequential()

# Step 1: Convolution
# classifier.add(Convolution2D(32(no of convolution), 3(no of row), 3(no of column)
#                , input_shape=(64, 64, 3), activation='relu'))

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# 32- feature detector into 3*3 matrix
# reshape transform all image into a single format

# Step 2: Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full Connection

classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("All Step Done")

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=8000,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=2000)