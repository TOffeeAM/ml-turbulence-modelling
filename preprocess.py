import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fluidfoam import readscalar, readsymmtensor, readtensor, readvector


def orderofMag(number):
    return math.floor(math.log(number, 10))


def magnitude(matrix):
    return (matrix**2).sum(axis=1)


def symm_tensor_convert(tensor):
    output_tensor = np.empty((tensor.shape[0], 9), dtype=float)
    output_tensor[:, 0:3] = tensor[:, 0:3]
    output_tensor[:, 3] = tensor[:, 1]
    output_tensor[:, 4] = tensor[:, 3]
    output_tensor[:, 5] = tensor[:, 4]
    output_tensor[:, 6] = tensor[:, 2]
    output_tensor[:, 7] = tensor[:, 4]
    output_tensor[:, 8] = tensor[:, 5]
    return output_tensor


def variable_size(type):
    if type == "scalar":
        return 1
    elif type == "vector":
        return 3
    elif type == "symmetry_tensor":
        return 6
    elif type == "tensor":
        return 9


def main(
    root_path="./Data",
    source_dir="Heat Exchanger 1",
    type="LES",
    reynolds_number="2243",
    high_fidelity="High_Fedility",
    frozen="Frozen_K_Omega",
):
    variable_names = {
        "k": {"type": "scalar", "folder": frozen},
        "omega": {"type": "scalar", "folder": frozen},
        "UMean": {"type": "vector", "folder": high_fidelity},
        "strainRateMean": {"type": "symmetric-tensor", "folder": frozen},
        "omegaTensorMean": {"type": "asymmetric-tensor", "folder": frozen},
        "WallDistance": {"type": "scalar", "folder": frozen},
        "ReTotMean": {"type": "tensor", "folder": high_fidelity},
    }

    data_dict = {}

    # Step 1 - Get list of all simulations
    source_path = os.path.join(root_path, source_dir, type, reynolds_number, "raw")
    target_path = os.path.join(root_path, source_dir, type, reynolds_number, "preprocessed")

    # Step 2 - Loop through all field variables
    dataset_size = 1
    folder_name = ""

    for name in variable_names:
        folder_name = variable_names[name]["folder"]

        if variable_names[name]["type"] == "scalar":
            data_read = readscalar(source_path, folder_name, name)

        elif variable_names[name]["type"] == "vector":
            data_read = readvector(source_path, folder_name, name).T

        elif variable_names[name]["type"] == "symmetric-tensor":
            data_read = readsymmtensor(source_path, folder_name, name).T

        elif variable_names[name]["type"] == "tensor":
            data_read = readtensor(source_path, folder_name, name).T

        elif variable_names[name]["type"] == "asymmetric-tensor":
            data_read = readtensor(source_path, folder_name, name).T

        dataset_size_new = data_read.shape[0]
        data_dict[name] = data_read
        data_read = 0

        if dataset_size > 0 and dataset_size != dataset_size_new:
            assert f"Inconsistent data load. Variable {name} has an different number of cells"
        dataset_size = dataset_size_new

    Neural_Network_dict = {}
    UMean = data_dict["UMean"]
    k = data_dict["k"]
    UMean_mag = magnitude(UMean)
    T_I = (1 / UMean_mag) * np.sqrt(2 * k / 3)
    Neural_Network_dict["T_I"] = T_I
    Neural_Network_dict["k"] = k

    print("Calculating local reynolds number.....", "\n")
    nu = input("Insert value kinematic viscosity: ")
    nu = float(nu)
    wall_dist = data_dict["WallDistance"]
    R_T = (wall_dist * np.sqrt(k)) / nu
    Neural_Network_dict["R_T"] = R_T

    srm = data_dict["strainRateMean"]
    srm_nine_element = symm_tensor_convert(srm)
    omega = data_dict["omega"]
    ones_matrix = np.ones((srm.shape[0], 1)) * np.identity(3).flatten()
    s_kk = srm[:, 0] + srm[:, 3] + srm[:, 5]
    s_kk = ones_matrix * s_kk.reshape(-1, 1)
    smn = (srm_nine_element - (s_kk / 3)) / omega.reshape(-1, 1)
    V_ij = smn

    I_1 = magnitude(smn)
    Neural_Network_dict["I_1"] = I_1

    omega_tens = data_dict["omegaTensorMean"]
    wmn = omega_tens / omega.reshape(-1, 1)
    I_2 = magnitude(wmn)
    Neural_Network_dict["I_2"] = I_2

    V_R = k / (omega * nu)
    Neural_Network_dict["V_R"] = V_R

    tensor_names = ["V_ij", "ReTotMean"]
    data_dict["V_ij"] = V_ij

    for name in tensor_names:
        for i in range(9):
            colmn_name = name + str(i)
            Neural_Network_dict[colmn_name] = data_dict[name][:, i]

    Neural_Network_df = pd.DataFrame(Neural_Network_dict)
    Neural_Network_data_path = os.path.join(target_path, "Neural_Network_data.csv")
    Neural_Network_df.to_csv(Neural_Network_data_path)


if __name__ == "__main__":
    main()
