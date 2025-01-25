import tensorflow as tf

# Ensure the model path is correct
model = tf.keras.models.load_model('artifacts/model.keras')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
with open('artifacts/model.tflite', 'wb') as f:
    f.write(tflite_model)
