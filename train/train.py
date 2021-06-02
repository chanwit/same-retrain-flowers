from kfp.components import OutputPath, InputPath


def train_model(
        train_data: InputPath(),
        shared_dir: str,
        mlpipeline_metrics_path: OutputPath('Metrics'),
        batch_size: int = 32,
        epochs: int = 5,
        model_name: str = "inception_v3",
        revision: str = "HEAD",
):
    model_handle_map = {
        "efficientnet_b0": "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
        "efficientnet_b1": "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1",
        "efficientnet_b2": "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1",
        "efficientnet_b3": "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1",
        "efficientnet_b4": "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1",
        "efficientnet_b5": "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1",
        "efficientnet_b6": "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1",
        "efficientnet_b7": "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1",
        "bit_s-r50x1": "https://tfhub.dev/google/bit/s-r50x1/1",
        "inception_v3": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
        "inception_resnet_v2": "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature-vector/4",
        "resnet_v1_50": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature-vector/4",
        "resnet_v1_101": "https://tfhub.dev/google/imagenet/resnet_v1_101/feature-vector/4",
        "resnet_v1_152": "https://tfhub.dev/google/imagenet/resnet_v1_152/feature-vector/4",
        "resnet_v2_50": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature-vector/4",
        "resnet_v2_101": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature-vector/4",
        "resnet_v2_152": "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/5",
        "nasnet_large": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/4",
        "nasnet_mobile": "https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4",
        "pnasnet_large": "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/4",
        "mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
        "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
        "mobilenet_v2_140_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
        "mobilenet_v3_small_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
        "mobilenet_v3_small_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
        "mobilenet_v3_large_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
        "mobilenet_v3_large_075_224": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
    }

    model_image_size_map = {
        "efficientnet_b0": 224,
        "efficientnet_b1": 240,
        "efficientnet_b2": 260,
        "efficientnet_b3": 300,
        "efficientnet_b4": 380,
        "efficientnet_b5": 456,
        "efficientnet_b6": 528,
        "efficientnet_b7": 600,
        "inception_v3": 299,
        "inception_resnet_v2": 299,
        "nasnet_large": 331,
        "pnasnet_large": 331,
    }

    model_handle = model_handle_map.get(model_name)
    pixels = model_image_size_map.get(model_name, 224)
    image_size = (pixels, pixels)
    print(f"Input size: {image_size}")
    # print(f"Cache Dir:  {cache_dir}")

    # cache_dir = None
    # import os
    # if 'KERAS_HOME' in os.environ:
    #   cache_dir = os.environ.get('KERAS_HOME')

    import tensorflow as tf
    import tensorflow_hub as hub

    data_file_path = str(train_data)
    data_dir = tf.keras.utils.get_file(
        'flower_photos',
        'file://' + data_file_path + '/flower_photos.tgz',
        untar=True)
#        cache_dir=cache_dir)

    print("Data Cached.")
    print(f"Data Dir:  {data_dir}")

    datagen_kwargs = dict(rescale=1. / 255, validation_split=.20)
    dataflow_kwargs = dict(target_size=image_size, batch_size=batch_size) # , interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

    train_datagen = valid_datagen

    train_generator = train_datagen.flow_from_directory(
        data_dir, subset="training", shuffle=True, **dataflow_kwargs)

    # ## Defining the model
    #
    # All it takes is to put a linear classifier on top of the
    # `feature_extractor_layer` with the Hub module.

    # For speed, we start out with a non-trainable `feature_extractor_layer`, but
    # you can also enable fine-tuning for greater accuracy.

    do_fine_tuning = True

    print("Building model with", model_handle)
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=image_size + (3,)),
        hub.KerasLayer(model_handle, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(train_generator.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + image_size + (3,))
    model.summary()

    ## Training the model

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    hist = model.fit(
        train_generator,
        epochs=epochs, steps_per_epoch=steps_per_epoch, verbose='2',
        validation_data=valid_generator,
        validation_steps=validation_steps).history

    accuracy = hist['accuracy'][-1]
    val_accuracy = hist["val_accuracy"][-1]
    print("accuracy:     {}".format(accuracy))
    print("val_accuracy: {}".format(val_accuracy))

    with open(f"{shared_dir}/val_accuracy_{model_name}", 'w') as writer:
        writer.write("%f\n" % val_accuracy)

    # TODO save accuracy
    saved_model_path = f"{shared_dir}/model_{model_name}_{revision}"
    print("Saving model to {0}".format(saved_model_path))
    # tf.saved_model.save(model, saved_model_path)
    tf.keras.models.save_model(model, saved_model_path)
