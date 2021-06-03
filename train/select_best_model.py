from kfp.components import InputPath
from typing import List, Dict


def select_best_model(
        shared_dir: str,
        model_names: List[str],
        revision: str,
):
    # this is a workaround for a bug when passing List[str] via pipeline
    # the list was serialized as a string, so we need to re-parse it
    import ast
    model_names = ast.literal_eval(model_names)

    val_accuracy: Dict[str, float] = {}
    for model_name in model_names:
        # print(f"{shared_dir}/val_accuracy_{model_name}")
        with open(shared_dir + "/val_accuracy_" + model_name, 'r') as reader:
            value = reader.readline()
            val_accuracy[model_name] = float(value)

    import pandas
    # df = pandas.DataFrame.from_dict(val_accuracy, index=[0])
    df = pandas.Series(val_accuracy)
    print(df)

    print("\n")

    best_model_name = max(val_accuracy, key=val_accuracy.get)
    print(f"Best Model: {best_model_name}")
    model_path: str = f"{shared_dir}/model_{best_model_name}_{revision}"
    bucket_path: str = f"mlpipeline/models/{best_model_name}/{revision}/"

    from kubernetes import client, config
    import base64
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    secret = v1.read_namespaced_secret("mlpipeline-minio-artifact", "kubeflow").data
    accesskey = base64.b64decode(secret["accesskey"]).decode("utf-8")
    secretkey = base64.b64decode(secret["secretkey"]).decode("utf-8")

    import s3fs
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=accesskey,
        secret=secretkey,
        client_kwargs={
            'endpoint_url': 'http://minio-service.kubeflow:9000'
        }
    )
    s3.put(lpath=model_path, rpath=bucket_path, recursive=True)
    with open(f"{shared_dir}/{revision}", "w") as file:
        file.write(best_model_name)
    s3.put(lpath=f"{shared_dir}/{revision}", rpath=f"mlpipeline/models/metadata/{revision}")

    # Save the model using keras export_saved_model function.
    # Note that specifically for TF-Serve,
    # the output directory should be structure as model_name/model_version/saved_model.
    # tf.keras.experimental.export_saved_model(model, model_version_path)

    #
    # expect that kubeflow already there in the deploy cluster
    #
    deploy = f"""
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: retrain-ui
  name: retrain-ui
  namespace: kubeflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: retrain-ui
  template:
    metadata:
      labels:
        app: retrain-ui
    spec:
      containers:
      - name: app
        image: quay.io/chanwit/retrain-demo-app:latest
        command: ["/usr/local/bin/streamlit"]
        args: ["run", "app.py"]
        env:
        - name: MODEL_NAME
          value: {best_model_name}
        - name: MODEL_REVISION
          value: {revision}
        - name: MODEL_ENDPOINT_URL
          value: http://f47dc4f41d8ce51e6cfc.southeastasia.cloudapp.azure.com:9000    
"""
