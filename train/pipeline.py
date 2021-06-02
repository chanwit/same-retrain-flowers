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
def tf2_retrain(epochs=5, batch_size=32, revision='HEAD'):
    """Pipeline steps"""

    # environment = "local"
    # model_names = ["mobilenet_v3_small_100_224"]

    environment = "azure"
    model_names = ["mobilenet_v3_small_100_224", "inception_v3", "efficientnet_b3"]

    pvc_name = "same-retrain-pvc"
    if environment == "azure":
        vop = dsl.VolumeOp(
            name=pvc_name,
            resource_name=pvc_name,
            size="10Gi",
            storage_class="azurefile",
            modes=dsl.VOLUME_MODE_RWM
        )
    else:
        vop = dsl.VolumeOp(
            name=pvc_name,
            resource_name=pvc_name,
            size="10Gi",
            storage_class="standard",
            modes=dsl.VOLUME_MODE_RWO
        )

    operations: Dict[str, dsl.ContainerOp] = dict()

    # preprocess
    import preprocess
    preprocess_factory = components.func_to_container_op(
        func=preprocess.preprocess_data,
        base_image="tensorflow/tensorflow:2.5.0",
        packages_to_install=['tensorflow-hub==0.12.0', 'matplotlib==3.2.2', 'scipy==1.4.1', 'Pillow==7.1.2']
    )
    operations['preprocess'] = preprocess_factory()
    # operations['preprocess'].add_pvolumes({"/cache": vop.volume})
    # operations['preprocess'].container.add_env_variable(k8s.V1EnvVar(name='KERAS_HOME', value='/keras'))
    # operations['preprocess'].execution_options.caching_strategy.max_cache_staleness = "P0D"

    # train
    import train
    train_factory = components.func_to_container_op(
        func=train.train_model,
        base_image="tensorflow/tensorflow:2.5.0",
        packages_to_install=['tensorflow-hub==0.12.0', 'matplotlib==3.2.2', 'scipy==1.4.1', 'Pillow==7.1.2']
    )

    for model_name in model_names:
        operations[model_name] = train_factory(
            train_data=operations['preprocess'].output,
            shared_dir="/shared",
            batch_size=batch_size,
            epochs=epochs,
            model_name=model_name,
            revision=revision,
        )
        operations[model_name].display_name = "Retrain {}".format(model_name)
        operations[model_name].add_pvolumes({"/shared": vop.volume})
        operations[model_name].after(operations['preprocess'])

    # select best model
    import select_best_model
    select_best_model_factory = components.func_to_container_op(
        func=select_best_model.select_best_model,
        base_image="tensorflow/tensorflow:2.5.0",
        packages_to_install=['tensorflow-hub==0.12.0', 'matplotlib==3.2.2', 'scipy==1.4.1', 'Pillow==7.1.2', 'kubernetes', 'pandas', 's3fs']
    )
    operations["select_best_model"] = select_best_model_factory(
        shared_dir="/shared",
        model_names=model_names,  # it gets serialized as STR
        revision=revision,
    )
    # no cache
    operations["select_best_model"].execution_options.caching_strategy.max_cache_staleness = "P0D"
    operations["select_best_model"].add_pvolumes({"/shared": vop.volume})
    for model_name in model_names:
        operations["select_best_model"].after(operations[model_name])


if __name__ == '__main__':
    compiler.Compiler().compile(tf2_retrain, __file__ + '.tar.gz')
