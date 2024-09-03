# Includes

import tensorflow as tf
import numpy as np

# Further imports are NOT allowed, please use the APIs in `tf`, `tf.keras` and `tf.keras.layers`!


def create_micro_kws_student_model(model_settings: dict, model_name: str = "micro_kws_student") -> tf.keras.Model:
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # First convolution
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=4,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=3,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=2,
        kernel_size=(4, 4),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=2,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="SAME",
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
   
    # Flatten for fully connected layers.
    x = tf.keras.layers.Flatten()(x)
    
    # Output fully connected.
    x = tf.keras.layers.Dense(units=model_settings["label_count"]*3, activation="relu")(x)
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)
    return tf.keras.Model(inputs, output, name=model_name)
