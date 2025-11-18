
#############################################################################################################
# Vision Transformer model with PyTorch.
#############################################################################################################


import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import socket
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm, trange

from model import ViT
from utils import get_device, get_positional_encodings


def test_visualize_positional_encodings():

    import matplotlib.pyplot as plt
    plt.imshow(get_positional_encodings(100, 300), cmap="hot", interpolation="nearest")
    plt.show()


def test_dummy():

    device = get_device()

    # Current model
    model = ViT(
        chw=(1, 28, 28),
        num_patches=7,
        num_blocks=2,
        hidden_dim=8,
        num_heads=2,
        out_dim=10
    ).to(device)

    x = torch.randn(7, 1, 28, 28) # Dummy images
    res = model(x)

    print("Result:", res.shape)  # torch.Size([7, 10])


def run(use_mlflow: bool = False):
    """
    Downloads MNIST dataset and train a basic Vision Transormer model with it.
    """

    print("\t=====\n\tSTART\n\t=====\n")

    if use_mlflow:
        experiment_name = "VIT_on_MNIST"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Error setting up MLflow experiment: {e}")
            return

    # hyper-parameters
    N_EPOCHS = 35
    LR = 0.01  # def 0.001

    # Loading MNIST data and converting it to tensors and
    # scale the values accordingly.
    transform = ToTensor()

    train_set = MNIST(root='data', train=True, download=True, transform=transform)
    test_set = MNIST(root='data', train=False, download=True, transform=transform)

    train_set , val_set = torch.utils.data.random_split(
        train_set, [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    # Debug: Reduce the train_set and val_set by using the first X samples.
    if False:
        train_set = torch.utils.data.Subset(train_set, torch.arange(5012))
        val_set = torch.utils.data.Subset(val_set, torch.arange(1024))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=64)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and hyperparameters.
    device = get_device()
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "\n"
    print(f"  Using device: {device_name}\n")

    # Construct the Vision Transformer model.
    number_of_patches = 7
    num_blocks = 2
    number_of_heads = 4  # 2?
    hidden_dim = 16 # 8?
    model = ViT(
        chw=(1, 28, 28),
        num_patches=number_of_patches,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        num_heads=number_of_heads,
        out_dim=10
    ).to(device)

    # Training loop.
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    criterion = CrossEntropyLoss()

    # Reduce LR when validation loss stops improving
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, threshold_mode='abs', min_lr=0.0001
    )
    print(f"Base LR: {LR:.4f}")

    for epoch in trange(N_EPOCHS, desc="Training"):

        # Training block.
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation block.
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)

                val_loss += loss.detach().cpu().item() / len(val_loader)

                val_correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                val_total += len(x)

        val_acc = (val_correct / val_total) * 100
        print(f"Epoch {epoch + 1}/{N_EPOCHS} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if LR != current_lr:
            print(f"Current LR: {current_lr:.4f}")
            LR = current_lr

    # Test.
    with torch.no_grad():
        correct, total = 0, 0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print("--------------")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


    # Log and save.
    dt = datetime.datetime.now().strftime('%Y%m%d')
    model_name = (
        "model" +
        "_np" + str(number_of_patches) +
        "_nh" + str(number_of_heads) +
        "_hd" + str(hidden_dim) +
        "_ep" + str(N_EPOCHS) + "-" + dt
    )

    # 1. MLFlow logging (params and metrics).
    if use_mlflow:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("num_of_patches", number_of_patches)
            mlflow.log_param("num_of_heads", number_of_heads)
            mlflow.log_param("num_of_epochs", N_EPOCHS)
            mlflow.log_param("LR", LR)
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("test_accuracy", correct / total * 100)

            # Log model.
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name=model_name,
                # registered_model_name="some_nice_name?"
            )

    # 2. Save model to disk.
    out_path = "output"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    model_name = out_path + "/" + model_name + ".ckpt"
    torch.save(model.state_dict(), model_name)

    print("\t========\n\tFINISHED\n\t========\n")


def log_local_model(model_path: str, ml_model_name: str, ml_experiment_name: str):
    """
    Log a local model file to an MLflow experiment.
    An automatic nice name is generated fir the model.
    """

    try:
        experiment = mlflow.get_experiment_by_name(ml_experiment_name)
        if experiment is None:
            mlflow.create_experiment(ml_experiment_name)
        mlflow.set_experiment(ml_experiment_name)
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        return

    state_dict = torch.load(model_path, map_location="cuda:0")

    # Init the model properly!
    number_of_patches = 7
    num_blocks = 2
    number_of_heads = 4  # 2?
    hidden_dim = 16 # 8?
    model = ViT(
        chw=(1, 28, 28),
        num_patches=number_of_patches,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        num_heads=number_of_heads,
        out_dim=10
    ).to("cuda:0")

    model.load_state_dict(state_dict)
    model.eval()

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=ml_model_name,
            registered_model_name=ml_model_name
        )


def register_local_model(r_id: str, model_name: str):
    """
    Register the model in MLflow.
    The model must have been logged previously.

    r_id : str
        The run ID (URI) of the referred model in the MLmodel directory.
    model_name : str
        Name of the registered model under which to create a new model version.
    """

    model_uri = f"runs:/{r_id}/{model_name}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print("Model registered as version:", result.version)


def load(r_id: str, model_name: str):
    """
    Load the model from the registry in MLflow.
    """

    test_set = MNIST(root='j:/AI_DATA', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    model_version = 1
    model_uri = f"models:/{model_name}/{model_version}"

    loaded_model = mlflow.pytorch.load_model(model_uri)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        correct, total = 0, 0

        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = loaded_model(x)
            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == '__main__':


    # Visualize a sine positional encoding formula.
    # test_visualize_positional_encodings()

    # Test with dummy data.
    # test_dummy()
    # sys.exit(0)

    use_mlflow = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", 5000)) != 0:
            print("MLflow server is not running on localhost:5000. Start it first.")
            sys.exit(1)
        else:
            print("MLflow server is running.")
            use_mlflow = True

    if use_mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  #######

    # top_models: pd.DataFrame = mlflow.search_logged_models(
    #     experiment_ids=["463343728754736413"],
    #     max_results=5,
    # )
    # new_df = top_models.iloc[:, [7, 9, 10]]
    # print(new_df)

    #################
    # Train and test.
    #################

    np.random.seed(0)
    torch.manual_seed(0)

    run(use_mlflow)
    sys.exit(0)


    ################################################
    # Load and register model subsequently.
    # If it was not done at the end of the training.
    ################################################

    # Log a local model.
    # log_local_model(
    #     model_path="output/model_np7_nh4_hd16_ep35-20251114.ckpt",
    #     ml_model_name="model_np7_nh4_hd16_ep35-20251114",
    #     ml_experiment_name="VIT_on_MNIST"
    # )
    # Register.
    # register_local_model(
    #     r_id="99148f8be0ad430f8edd5b874fdf02be",
    #     model_name="model_np7_nh4_hd16_ep35-20251114"
    # )

    # Load and run the model.
    # load("", "model_np7_nh2_ep60-20250701")  ########
