import tensorflow as tf
from ssl_models import MultimediaSSLModel
from data_loader import MultimediaDataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load and preprocess the dataset
dataset = MultimediaDataset('path_to_dataset')
train_data, val_data, test_data = dataset.split_train_val_test()

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'path_to_train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'path_to_val_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'path_to_test_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Initialize and compile the SSL model
ssl_model = MultimediaSSLModel(input_shape=(150, 150, 3))
ssl_model.compile(optimizer=Adam(lr=1e-4), loss='contrastive_loss')

# Train the SSL model
ssl_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate the model
metrics = ssl_model.evaluate(test_generator)
print("Evaluation Metrics:", metrics)

# Save the model
ssl_model.save('ssl_multimedia_recommendation_model.h5')
