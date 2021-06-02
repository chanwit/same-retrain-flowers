from kfp.components import OutputPath


def preprocess_data(
		output_path: OutputPath(str)
	):
	import requests

	# Load raw data
	train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
	print('Downloading dataset from {0} to {1}'.format(train_dataset_url, output_path))
	data = requests.get(train_dataset_url).content

	import os
	import shutil
	shutil.rmtree(output_path, ignore_errors=True)
	os.mkdir(output_path)

	# shutil.rmtree(cache_dir + "/datasets/flower_photos", ignore_errors=True)

	with open(output_path + '/flower_photos.tgz', 'wb') as writer:
		writer.write(data)

	print("Downloaded.")

	# import tensorflow as tf
	# tf.keras.utils.get_file(
	#	'flower_photos',
	#	'file://' + output_path + '/flower_photos.tgz',
	#	untar=True,
	#	cache_dir=cache_dir)
	# print("Cached.")
