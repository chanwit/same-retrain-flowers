from kfp.components import OutputPath

def preprocess_data(
		output_path: OutputPath(str)
	):
	import requests
	from pathlib import Path

	# Load raw data
	train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
	print('Downloading dataset from {0} to {1}'.format(train_dataset_url, output_path))
	data = requests.get(train_dataset_url).content

	import os
	os.mkdir(output_path)

	with open(output_path + '/flower_photos.tgz', 'wb') as writer:
		writer.write(data)

	print("Done.")