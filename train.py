from kfp.components import OutputPath, InputPath

def train_model(
		train_data: InputPath(),
		mlpipeline_metrics_path: OutputPath('Metrics'),
		output_path: OutputPath(str),
		batch_size: int = 32,
		epochs: int = 5,
	):

	model_name = "mobilenet_v3_small_100_224"
	model_handle = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5"
	pixels = 224

	IMAGE_SIZE = (pixels, pixels)
	print(f"Input size {IMAGE_SIZE}")

	cache_dir = None

	import os

	if 'KERAS_HOME' in os.environ:
		cache_dir = os.environ.get('KERAS_HOME')


	import itertools
	import numpy as np
	import tensorflow as tf
	import tensorflow_hub as hub

	data_file_path = str(train_data)
	data_dir = tf.keras.utils.get_file(
		'flower_photos',
		'file://' + data_file_path + '/flower_photos.tgz',
		untar=True,
		cache_dir=cache_dir)

	print("Data Cached.")

	datagen_kwargs = dict(rescale=1./255, validation_split=.20)
	dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=batch_size, interpolation="bilinear")

	valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		**datagen_kwargs)
	valid_generator = valid_datagen.flow_from_directory(
		data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

	do_data_augmentation = False

	if do_data_augmentation:
		train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
			rotation_range=40,
			horizontal_flip=True,
			width_shift_range=0.2, height_shift_range=0.2,
			shear_range=0.2, zoom_range=0.2,
			**datagen_kwargs)
	else:
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
		tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
		hub.KerasLayer(model_handle, trainable=do_fine_tuning),
		tf.keras.layers.Dropout(rate=0.2),
		tf.keras.layers.Dense(train_generator.num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
	])
	model.build((None,)+IMAGE_SIZE+(3,))
	model.summary()


	## Training the model

	model.compile(
		optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
		loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
		metrics=['accuracy'])


	steps_per_epoch = train_generator.samples // train_generator.batch_size
	validation_steps = valid_generator.samples // valid_generator.batch_size
	model.fit(
		train_generator,
		epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=2,
		validation_data=valid_generator,
		validation_steps=validation_steps).history

	print("Saving model to {0}".format(output_path))
	tf.saved_model.save(model, output_path)
