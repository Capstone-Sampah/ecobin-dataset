import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Get the list of physical devices (CPUs and GPUs)
physical_devices = tf.config.list_physical_devices()
if len(physical_devices) == 0:
    print("No devices found.")
else:
    for device in physical_devices:
        print(f"Device name: {device.name}, type: {device.device_type}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
