"""Main pipeline file"""

from typing import Dict
import kfp.dsl as dsl
import kfp.compiler as compiler
import kfp.components as components
from kubernetes import client as k8s

@dsl.pipeline(
	name="TF2 Flowers retrain",
	description="TF2 Flowers retrain"
)
def tf2_retrain(epochs=5, batch_size=32):
	"""Pipeline steps"""

	vop = dsl.VolumeOp(
		name="tf2-retrain-pvc",
		resource_name="tf2-retrain-pvc",
		size="10Gi",
		storage_class="azurefile",
		modes=dsl.VOLUME_MODE_RWM
	)

	operations: Dict[str, dsl.ContainerOp] = dict()

	# preprocess
	import preprocess
	preprocessFactory = components.func_to_container_op(
			func=preprocess.preprocess_data,
			base_image="tensorflow/tensorflow:2.4.1",
			packages_to_install=[]
	)
	operations['preprocess'] = preprocessFactory()
	operations['preprocess'].add_pvolumes({"/keras": vop.volume})
	operations['preprocess'].container.add_env_variable(k8s.V1EnvVar(name='KERAS_HOME', value='/keras'))

	# train
	import train
	trainFactory = components.func_to_container_op(
			func=train.train_model,
			base_image="tensorflow/tensorflow:2.4.1",
			packages_to_install=['tensorflow_hub', 'matplotlib', 'scipy', 'Pillow']
		)
	operations['train'] = trainFactory(
			train_data=operations['preprocess'].output,
			batch_size=batch_size,
			epochs=epochs
		)
	operations['train'].add_pvolumes({"/keras": vop.volume})
	operations['train'].container.add_env_variable(k8s.V1EnvVar(name='KERAS_HOME', value='/keras'))
	operations['train'].after(operations['preprocess'])

if __name__ == '__main__':
	compiler.Compiler().compile(tf2_retrain, __file__ + '.tar.gz')