import torch
import torch.optim as optim
import math
import cmath
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from FNN import FNN
import os
from scipy.interpolate import griddata
import json


def plot_graphs(
    model_path,
    model_id,
    train_loss_data,
    train_norm_loss_list,
    train_norm_loss_by_component_list,
    train_loss_by_component_list,
    epoch,
    epoch_list,
    c1_all,
    input_feature_list,
    input_feature_1_all,
    input_feature_2_all,
    resolution_grid=100,
):
    """
    Plottinig function to understand c1 mapping and losses
    """
    # Step 1 - Plot overalll loss
    plt.plot(epoch_list[: epoch + 1], train_loss_data, "o", color="blue", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(model_path, "training_loss.png"))
    plt.close()

    # Step 2 - Plot component normalised losses
    for type_loss in ["mse", "normalised"]:
        fig, ax = plt.subplots(2, 4, figsize=(15, 5), constrained_layout=True, sharex=False, sharey=False)
        train_loss_list = (
            train_norm_loss_by_component_list if type_loss == "normalised" else train_loss_by_component_list
        )
        train_loss_list_np = np.array(train_loss_list).reshape(-1, 6)
        plots = [
            # {"data": train_norm_loss_list, "label": "Component Average"},
            {"data": train_loss_list_np[:, 0], "label": "xx"},
            {"data": train_loss_list_np[:, 1], "label": "xy"},
            {"data": train_loss_list_np[:, 2], "label": "xz"},
            {"data": train_loss_list_np[:, 3], "label": "yy"},
            {"data": train_loss_list_np[:, 4], "label": "yz"},
            {"data": train_loss_list_np[:, 5], "label": "zz"},
            {"data": train_loss_list_np.mean(axis=1), "label": "mean"},
            {"data": train_loss_list_np.sum(axis=1), "label": "sum"},
        ]
        i = 0

        for row in ax:
            for col in row:
                col.plot(epoch_list[: epoch + 1], plots[i]["data"], "o", color="blue", label="Training Loss")
                col.set_xlabel("Number of Epochs")
                col.set_ylabel("Relative Error " + type_loss + " " + plots[i]["label"])
                i += 1

        plt.savefig(os.path.join(model_path, f"training_{type_loss}_loss.png"))
        plt.close()

        # Step 3 - Plot c1 output
        for counter, input_feature_all in enumerate([input_feature_1_all, input_feature_2_all]):
            plt.plot(input_feature_all, c1_all, "o", color="blue", label="C1 Outputs")
            plt.xlabel(input_feature_list[counter])
            plt.ylabel("C1")
            plt.savefig(os.path.join(model_path, f"c1vs{input_feature_list[counter]}.png"))
            plt.close()

        # Step 4 - Plot c1 historgam
        plt.hist(input_feature_1_all, density=True, bins=30)
        plt.xlabel("C1")
        plt.ylabel("Density")
        plt.savefig(os.path.join(model_path, "c1_hist.png"))
        plt.close()

        # Step 5 - Contour plot
        input_feature_1_all_np = np.array(input_feature_1_all)
        input_feature_2_all_np = np.array(input_feature_2_all)
        c1_all_np = np.array(c1_all).flatten()
        x = np.linspace(input_feature_1_all_np.min(), input_feature_1_all_np.max(), resolution_grid)
        y = np.linspace(input_feature_2_all_np.min(), input_feature_2_all_np.max(), resolution_grid)
        X, Y = np.meshgrid(x, y)
        Z = griddata((input_feature_1_all_np, input_feature_2_all_np), c1_all_np, (X, Y), method="nearest")

        with open(os.path.join(model_path, "mapping0.npy"), "wb") as f:
            np.save(f, X)
            np.save(f, Y)
            np.save(f, Z)

        plt.contourf(X, Y, Z, 12)
        plt.colorbar()
        plt.xlabel(input_feature_list[0])
        plt.ylabel(input_feature_list[1])
        plt.title(f"C1 model{str(model_id)}")
        plt.savefig(os.path.join(model_path, "c1_contour.png"))
        plt.close()
