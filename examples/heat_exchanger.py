#!/usr/bin/env python3

from pathlib import Path

from turbulence_model.data_filtering import main as data_filtering
from turbulence_model.preprocess import main as preprocess
from turbulence_model.train import main as train

ROOT_DIR = str(Path(__file__).parent.parent.resolve() / "Data")
DATA_DIR = "Heat Exchanger 1"

# Preprocess data
preprocess(
    root_path=ROOT_DIR,
    source_dir=DATA_DIR,
    type="LES",
    reynolds_number="2243",
    high_fidelity="High_Fedility",
    frozen="Frozen_K_Omega",
)

# Filter data
data_filtering(
    type="LES",
    Heat_Exchanger_type=DATA_DIR,
    High_Fedility_type="LES",
    Reynolds_Number="2243",
    cutoffs={"I_1": 0.95, "I_2": 0.95},
    input_feature_list=["V_R", "R_T"],
    k_threshold=0.1,
    data_breakdown_cutoff=True,
)

# Train model
train(
    learning_rate=5e-4,
    epochs=250,
    train_batch_size=4048,
    valid_batch_size=32,
    GPU=False,
    Heat_Exchanger_type=DATA_DIR,
    High_Fedility_type="LES",
    Reynolds_Number="2243",
    dataset_choice="20percent",
    verbose=False,
    data_cut_off=True,
    root_path=ROOT_DIR,
    ignore_diagonals=False,
    output_clamp_range={"min": None, "max": None},
    weight_decay=0,
    input_feature_list=["I_1", "I_2"],
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
)
