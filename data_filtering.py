from cmath import sqrt
import time
import dbm
import math
import numpy as np
import pandas as pd
import os
import torch
import inspect
import json
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms
from fluidfoam import readscalar, readvector, readsymmtensor, readtensor
from tqdm import tqdm
from scipy.stats import norm


def plot_histograms(data_set, postfix, source_path):
    for col in data_set.columns:
        fig = data_set.plot(y=col, kind="hist").get_figure()
        fig.savefig(os.path.join(source_path, f"{col}_{postfix}.png"))


def main(
    type="LES",
    Heat_Exchanger_type="Heat Exchanger 1",
    High_Fedility_type="LES",
    Reynolds_Number="2243",
    cutoffs={"I_1": 0.95, "I_2": 0.95},
    input_feature_list=["V_R", "R_T"],
    k_threshold=0.1,
    data_breakdown_cutoff=True,
):
    arguments = locals()
    with open(os.path.join("data_filtering_arguments.json"), "w") as fp:
        json.dump(arguments, fp)

    # Step 0 - Set Paths
    root_path = "./Data"
    source_path = os.path.join(
        root_path,
        Heat_Exchanger_type,
        High_Fedility_type,
        Reynolds_Number,
        "preprocessed",
    )

    print("The K threshold is: ", k_threshold)

    # Step 1 - Load data
    input_features_path = os.path.join(source_path, "Neural_Network_data.csv")
    data_set = pd.read_csv(input_features_path)
    data_set.drop(labels="Unnamed: 0", axis=1, inplace=True)
    input_feature_names = ["I_1", "I_2", "V_R", "R_T", "T_I"]
    print("The shape before filtering: ", data_set.shape)

    # Step 2 - Turbulent Kinetic Energy Filtering
    print("Shape before k filtering: ", data_set.shape)
    k_max = data_set["k"].max()
    k_min = data_set["k"].min()
    data_set["k"] = (data_set["k"] - k_min) / (k_max - k_min)
    data_set = data_set[data_set["k"] > k_threshold]
    data_set.reset_index(drop=True, inplace=True)
    data_set["k"] = (data_set["k"] * (k_max - k_min)) + k_min
    print("Shape after k filtering: ", data_set.shape)

    # Step 3 - Prefiltering histograms
    plot_histograms(data_set, "prefilter", source_path)

    # Step 4 - Filtering Tails of distributions
    print("Shape before filtering: ", data_set.shape)
    quantiles_dict = {}
    for key in cutoffs:
        quantiles_dict[key] = data_set[[key]].quantile(cutoffs[key])[0]
    quantiles = pd.Series(quantiles_dict)
    data_set = data_set[~((data_set[cutoffs.keys()] > (quantiles))).any(axis=1)]
    print("Shape after filtering: ", data_set.shape)
    data_set_size = data_set.shape[0]

    # Step 5 - Postfiltering histograms of all features
    plot_histograms(data_set, "postfilter", source_path)

    # Step 6 - Normalise input features
    i = 0
    for name in input_feature_names:
        max = (data_set[name]).max()
        min = (data_set[name]).min()
        data_set[name] = (data_set[name] - min) / (max - min)

    # Step 6 - Shuflle data
    shuffle_data = data_set.sample(frac=1)
    data_set = data_set.reindex(shuffle_data.index)

    # Step 7 - Remove unused data
    labels_to_drop = ["V_ij3", "V_ij6", "V_ij7", "ReTotMean3", "ReTotMean6", "ReTotMean7"]
    data_set.drop(labels=labels_to_drop, axis=1, inplace=True)

    # Step 8 - Create sub-datasets

    data_breakdown = [0.05, 0.2, 0.5, 1]

    for num in data_breakdown:
        dataset_size = int((data_set.shape[0]) * num)
        data_ = data_set[0:dataset_size]
        directory = os.path.join(source_path, f"{int(num*100)}%_Data")
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_file = os.path.join(directory, "data_set.csv")
        data_.to_csv(data_file)


if __name__ == "__main__":
    main()
