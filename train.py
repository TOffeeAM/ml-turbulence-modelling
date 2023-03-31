import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from FNN import FNN
import os
import json
from plot_graphs import plot_graphs


def normalized_loss(predicted, real):
    # Define Loss Function
    row_loss = abs(predicted - real)
    den_mag = abs(real)
    relative_error_all = torch.div(row_loss, den_mag)
    mean_error_all_components = relative_error_all.mean()
    mean_error_per_component = torch.mean(relative_error_all, dim=0)
    return mean_error_all_components, mean_error_per_component


def mse_loss_per_component(predicted_scaled, real_scaled):
    # Define Loss Function
    loss = nn.MSELoss(
        reduction="none",
    )
    square_loss_all = loss(predicted_scaled, real_scaled)
    return square_loss_all.mean(axis=0)


def terminal_prints(V_ij, cons_2pk, output, prediction, ReTotMean):
    print("V_ij brinted: ", "\n", V_ij[0])
    print("The kinetic energy term: ", "\n", cons_2pk[0:3])
    print("The ouptut of the NN: ", output[0:3])
    print("Predicton from first Batch: ", "\n", prediction[0:3])
    print("Target in the loop: ", "\n", ReTotMean[0:3])
    print("\n", "\n")


def weights(input):
    mean_per_component = input.mean(axis=0)
    alpha = torch.sqrt(abs(1 / mean_per_component))
    return alpha


class dds(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def main(
    learning_rate=5e-4,
    epochs=250,
    train_batch_size=4048,
    valid_batch_size=32,
    GPU=False,
    Heat_Exchanger_type="Heat Exchanger 1",
    High_Fedility_type="LES",
    Reynolds_Number="2243",
    dataset_choice="20percent",
    verbose=False,
    data_cut_off=True,
    root_path="./Data",
    ignore_diagonals=False,
    output_clamp_range={"min": None, "max": None},  # use {'min': None, 'max': None} for no clamp,
    weight_decay=0,  # L2 reguralisation e.g. 1e-5
    input_feature_list=["I_1", "I_2"],  # I_1,I_2,V_R,R_T,T_I
    output_feature_list=[
        "k",
        "V_ij0",
        "V_ij1",
        "V_ij2",
        "V_ij4",
        "V_ij5",
        "V_ij8",
        "ReTotMean0",
        "ReTotMean1",
        "ReTotMean2",
        "ReTotMean4",
        "ReTotMean5",
        "ReTotMean8",
    ],
    num_hidden_layers=4,
    num_neurons=[5, 5, 5, 5],
    model_id=0,
):
    # Step 0 - Create model folder:
    arguments = locals()
    model_path = f"./weights/model_{str(model_id)}_{Heat_Exchanger_type}_{Reynolds_Number}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(os.path.join(model_path, "train_arguments.json"), "w") as fp:
        json.dump(arguments, fp, indent=4)

    # GPU / CPU selection
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # Step 1 - Load Data
    source_path = os.path.join(root_path, Heat_Exchanger_type, High_Fedility_type, Reynolds_Number, "preprocessed")
    alpha, train_loader, valid_loader, input_train, label_train, data_file = load_data(
        source_path,
        dataset_choice,
        device,
        train_batch_size,
        valid_batch_size,
        input_feature_list,
        output_feature_list,
        data_cut_off,
    )
    # Optional - Set loss contribution of diagonal terms to zero
    if ignore_diagonals:
        alpha[[0, 3, 5]] = 0
    print("Weight alpha values: ", alpha)

    # Step 2 - Choose Model Architecture
    model = FNN(
        n_inputs=input_train.shape[1], n_outputs=1, num_hidden_layers=num_hidden_layers, num_neurons=num_neurons
    )

    # Step 3 - Define Loss and Optimiser
    nn_loss = torch.nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Step 4 - Constants
    rho = 1
    delta_ij = [0.3333, 0, 0, 0.3333, 0, 0.3333]
    delta_ij = (torch.tensor(delta_ij)).to(device)

    # Step 5 - Initialise arrays
    train_loss_data = []
    train_norm_loss_list = []
    train_norm_loss_by_component_list = []
    train_loss_by_component_list = []
    epoch_list = np.arange(start=0, stop=epochs, step=1)

    k_index = output_feature_list.index("k")
    Re_start_index = output_feature_list.index("ReTotMean0")
    Re_end_index = output_feature_list.index("ReTotMean8")
    V_ij_start_index = output_feature_list.index("V_ij0")
    V_ij_end_index = output_feature_list.index("V_ij8")

    for epoch in range(epochs):
        print(f"Start training for {epoch} epochs...")
        sum_train_loss = 0
        sum_norm_loss = 0
        sum_norm_loss_by_component = np.zeros((6))
        sum_loss_by_component = np.zeros((6))
        z = 0
        model.train().to(device)
        t0 = time.perf_counter()
        input_feature_1_all = []
        input_feature_2_all = []
        c1_all = []

        # Loop through train batches
        for features, targets in train_loader:
            # Step 6.1 - Get batch targets and scalings
            ReTotMean = targets[:, Re_start_index : Re_end_index + 1]
            cons_2pk = ((targets[:, k_index]) * 2 * rho).unsqueeze(1)
            V_ij = targets[:, V_ij_start_index : V_ij_end_index + 1]

            # Step 6.2 - Re-set the gradients
            optimizer.zero_grad()

            # Step 6.3 - Forward pass and predict Reynolds Stresses
            output = model.forward(features, num_neurons)

            if (output_clamp_range["min"] is not None) or (output_clamp_range["max"] is not None):
                output = output.clamp(min=output_clamp_range["min"], max=output_clamp_range["max"])
            prediction = cons_2pk * ((output * V_ij) + delta_ij)
            prediction_scaled = prediction * alpha
            ReTotMean_scaled = ReTotMean * alpha
            if verbose:
                print("*" * 50)
                print("output*vij", (output * V_ij)[0], " delta_ij:  ", (delta_ij))
                print("predicted : ", prediction[0])
                print("ground truth : ", ReTotMean[0])
                print("predicted scaled: ", prediction_scaled[0])
                print("ground truth scaled: ", ReTotMean_scaled[0])

            # Step 6.4 - Calculate loss
            loss = nn_loss(prediction_scaled, ReTotMean_scaled)
            norm_loss, norm_loss_per_component = normalized_loss(prediction, ReTotMean)
            loss_per_component = mse_loss_per_component(prediction_scaled, ReTotMean_scaled)

            # Step 6.5 - Backpropgation and optimiser step
            loss.backward()
            optimizer.step()

            # Step 6.6 - Store loss calculations
            sum_train_loss += loss.item()
            sum_norm_loss += norm_loss.item()
            sum_norm_loss_by_component = sum_norm_loss_by_component + norm_loss_per_component.cpu().detach().numpy()
            sum_loss_by_component = sum_loss_by_component + loss_per_component.cpu().detach().numpy()
            input_feature_1_all.extend(features[:, 0].cpu().detach().numpy().tolist())
            input_feature_2_all.extend(features[:, 1].cpu().detach().numpy().tolist())

            c1_all.extend(output.cpu().detach().numpy().tolist())

            z = z + 1

        # Step 7 - Calculate losses for epoch
        train_loss = sum_train_loss / z
        train_norm_loss = sum_norm_loss / z
        train_norm_loss_by_component = sum_norm_loss_by_component / z
        train_loss_by_component = sum_loss_by_component / z

        train_loss_data.append(train_loss)
        train_norm_loss_list.append(train_norm_loss)
        train_norm_loss_by_component_list.append(train_norm_loss_by_component)
        train_loss_by_component_list.append(train_loss_by_component)

        # Step 8 - Print terminal outputs
        if verbose:
            terminal_prints(V_ij, cons_2pk, output, prediction, ReTotMean)
        print(
            "Epoch: ",
            epoch,
            ", Training Loss: ",
            float(train_loss),
            "\t",
            "Training Normalized Loss: ",
            train_norm_loss,
            "\t",
        )

        # Step 9 - Plot graphs
        plot_graphs(
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
        )

        # Step 10 - Store model
        torch.save(model.state_dict(), os.path.join(model_path, "fnn.pt"))


def load_data(
    source_path,
    dataset_choice,
    device,
    train_batch_size,
    valid_batch_size,
    input_feature_list,
    output_feature_list,
    data_cut_off,
):
    """
    Load data from files and add to Dataloader
    """
    # Select Data
    if data_cut_off:
        print("The type: ", type(dataset_choice))
        num = int(dataset_choice.replace("percent", ""))
    else:
        data_breakdown = {
            "5percent": 5,
            "10percent": 10,
            "50percent": 50,
            "100percent": 100,
        }  # Percentage of Data to Train
        num = data_breakdown[dataset_choice]

    data_file = str(int(num)) + "%_Data"
    data_set = pd.read_csv(os.path.join(source_path, str(data_file), "data_set.csv"))
    data_set.drop(labels=["Unnamed: 0"], axis=1, inplace=True)  # Drop Automatic Column generated by read_csv
    data_set_size = data_set.shape[0]

    # Extract if needed I_1,I_2,V_R,R_T,T_I
    input_feature_names = ["I_1", "I_2", "V_R", "R_T", "T_I"]
    input_features = data_set[input_feature_list]
    labels = data_set[output_feature_list]

    # Calculation of Weights for MSE-Loss
    High_Fedility_ReTotMean = labels.loc[:, "ReTotMean0":"ReTotMean8"]
    High_Fedility_ReTotMean = torch.tensor(High_Fedility_ReTotMean.values, requires_grad=True).to(device)
    alpha = weights(High_Fedility_ReTotMean)
    alpha = alpha.clone().detach().requires_grad_(False)

    # Plot input features:
    for col in input_features.columns:
        fig = input_features.plot(y=col, kind="hist").get_figure()
        fig.savefig(f"./Data/{col}.png")

    # Split intro Train, Valid, and Test
    train_size = int(1 * data_set_size)

    # Training Data Set
    input_train = input_features[:train_size]
    label_train = labels[:train_size]

    # Validation Data Set
    input_valid = input_features[train_size:]
    label_valid = labels[train_size:]

    # Convert the Dataset into tensors
    input_train = torch.tensor(input_train.values, requires_grad=True).to(device)
    input_valid = torch.tensor(input_valid.values, requires_grad=True).to(device)

    label_train = torch.tensor(label_train.values, requires_grad=True).to(device)
    label_valid = torch.tensor(label_valid.values, requires_grad=True).to(device)

    input_train = input_train.float()
    input_valid = input_valid.float()

    label_train = label_train.float()
    label_valid = label_valid.float()

    # Define Data and Batch Size
    mytrain_data = dds(input_train, label_train)
    myvalid_data = dds(input_valid, label_valid)
    print("Dataset size: ", len(mytrain_data))

    train_loader = DataLoader(dataset=mytrain_data, batch_size=train_batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=myvalid_data, batch_size=valid_batch_size, shuffle=False)

    return alpha, train_loader, valid_loader, input_train, label_train, data_file


if __name__ == "__main__":
    main()
