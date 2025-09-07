import tensorflow as tf

# Load your existing Keras model
print("Loading original Keras model...")
model = tf.keras.models.load_model('crnn_model.h5')

# Create a TFLite converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- THIS IS THE FIX ---
# Enable a compatibility mode (Select TF Ops) to handle the special
# operations used by the GRU layers in your model.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TFLite ops.
    tf.lite.OpsSet.SELECT_TF_OPS    # Enable TensorFlow ops.
]
converter._experimental_lower_tensor_list_ops = False
# --- END OF FIX ---

# Perform the conversion
print("Converting model to TensorFlow Lite format...")
tflite_model = converter.convert()

# Save the new, smaller model to a file
with open('crunn_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model successfully converted and saved as crunn_model.tflite!")