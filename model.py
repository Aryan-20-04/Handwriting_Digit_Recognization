import tensorflow as tf
from keras import layers, models,callbacks
from keras._tf_keras.keras.datasets import mnist
import json

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint=callbacks.ModelCheckpoint('best_model.keras',monitor='val_loss',save_best_only=True,mode='min')
early_stop=callbacks.EarlyStopping(monitor='val_loss',patience=3)
# Train the model
history=model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test),callbacks=[checkpoint,early_stop])
with open('history.json','w') as f:
    json.dump(history.history,f)
# Save the model
model.save('mnist_digit_classifier.keras')
