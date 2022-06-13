import tensorflow as tf
from tensorflow import keras

# ======================================================================================================================


def geertnet_preprocessing(x):
    # Model requires range [0-1].
    return tf.image.convert_image_dtype(x, tf.float32)


def geertnet():
    layers = [
        keras.layers.Input(shape=(96, 96, 3)),
    ]

    layers.extend([
        keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
        keras.layers.Conv2D(16, (3, 3), padding='valid', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu'),
        keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu'),
        keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(2, activation='softmax'),
    ])

    model = keras.models.Sequential(layers)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    return model


# ======================================================================================================================


def resnet_preprocessing(x):
    x = tf.cast(x, tf.float32)

    # Model requires resnet preprocess input function.
    return keras.applications.resnet_v2.preprocess_input(x)


def resnet():
    layers = [
        keras.layers.Input(shape=(96, 96, 3)),
        keras.layers.ZeroPadding2D(padding=64),
    ]

    resnet_layer = keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        classes=2,
    )

    layers.append(resnet_layer)

    model = keras.models.Sequential(layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()],
    )

    return model


# ======================================================================================================================


def densenet_preprocessing(x):
    x = tf.cast(x, tf.float32)

    # Model requires densenet preprocess input function.
    return keras.applications.densenet.preprocess_input(x)


def densenet():
    layers = [
        keras.layers.Input(shape=(96, 96, 3)),
        keras.layers.ZeroPadding2D(padding=64),
    ]

    densenet_layer = keras.applications.DenseNet121(
        include_top=True,
        weights=None,
        classes=2,
    )

    layers.append(densenet_layer)

    model = keras.models.Sequential(layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()],
    )

    return model


# ======================================================================================================================


def efficientnet_preprocessing(x):
    # Models require range [0-255].
    return tf.cast(x, tf.float32)


def efficientnetb0():
    layers = [
        keras.layers.Input(shape=(96, 96, 3)),
        keras.layers.ZeroPadding2D(padding=64),
    ]

    efficientnet_layer = keras.applications.EfficientNetB0(
        include_top=True,
        weights=None,
        classes=2,
        drop_connect_rate=0.4
    )

    layers.append(efficientnet_layer)

    model = keras.models.Sequential(layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()],
    )

    return model


# ======================================================================================================================


_models = {
    'geertnet': (geertnet, geertnet_preprocessing),
    'resnet': (resnet, resnet_preprocessing),
    'densenet': (densenet, densenet_preprocessing),
    'efficientnetb0': (efficientnetb0, efficientnet_preprocessing),
}


def create(model_id):
    assert model_id in _models
    model, preprocessing = _models[model_id]
    return model(), preprocessing
