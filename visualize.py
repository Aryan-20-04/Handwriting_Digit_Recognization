import json
import matplotlib.pyplot as plt

# Load the training history from the JSON file
try:
    with open('history.json', 'r') as f:
        history = json.load(f)
except FileNotFoundError:
    raise Exception("The history.json file was not found. Please ensure it exists.")

# Create subplots for accuracy and loss
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plotting accuracy
axs[0].plot(history['accuracy'], label='Training Accuracy', color='blue')
axs[0].plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(loc='lower right')
axs[0].grid()

# Plotting loss
axs[1].plot(history['loss'], label='Training Loss', color='blue')
axs[1].plot(history['val_loss'], label='Validation Loss', color='orange')
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(loc='upper right')
axs[1].grid()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
