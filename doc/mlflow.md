

## Start MLFlow UI (from CLI)

mlflow ui

## Set ip address and port

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # localhost

## 1. Create (or retrieve) experiment
All trainings (called runs) belong to a particular experiment.
```
experiment_name = "VIT_on_MNIST"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Error setting up MLflow experiment: {e}")
    return
```
## 2. Log params, metrics and model
Define a model name (this is not the Run Name on the UI, which is generated automatically).
This run will have a unique Run ID and will be listed under the experiment that was defined in Step 1. 
```
model = model_object  # some nn.Module object in case of PyTorch
model_name = "my_model"

with mlflow.start_run(run_name=model_name):
    mlflow.log_param("num_of_patches", number_of_patches)
    mlflow.log_param("num_of_heads", number_of_heads)
    mlflow.log_param("num_of_epochs", N_EPOCHS)
    mlflow.log_param("LR", LR)
    mlflow.log_metric("train_loss", train_loss)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("test_accuracy", correct / total * 100)

    # Log model.
    mlflow.pytorch.log_model(model, model_name)
```