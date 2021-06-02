from kfp.components import InputPath
from typing import List


def select_best_model(
        shared_dir: str,
        model_names: List[str],
        revision: str,
):
    print(shared_dir)
    import ast
    model_names = ast.literal_eval(model_names)
    print(model_names)

    val_accuracy = {}

    for model_name in model_names:
        print(f"{shared_dir}/val_accuracy_{model_name}")
        with open(shared_dir + "/val_accuracy_" + model_name, 'r') as reader:
            value = reader.readline()
            val_accuracy[model_name] = float(value)
        model_path = f"{shared_dir}/model_{model_name}_{revision}"
        print(model_path)

    for model_name in model_names:
        print(f"{model_name} val acc: {val_accuracy[model_name]}")

    # choose best
    # push to bucket
    # Save the model using keras export_saved_model function.
    # Note that specifically for TF-Serve,
    # the output directory should be structure as model_name/model_version/saved_model.
    # tf.keras.experimental.export_saved_model(model, model_version_path)