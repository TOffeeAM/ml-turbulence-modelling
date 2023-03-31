import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json


class FNN(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, num_hidden_layers, num_neurons):
        super(FNN, self).__init__()
        self.inputs = n_inputs
        self.outputs = n_outputs
        self.num_neurons = num_neurons
        self.num_hiden_layers = num_hidden_layers

        # define the layers:
        size_ = len(num_neurons)
        for i in range(num_hidden_layers):
            hidden_layer = "fc" + str(i + 1)
            if i == 0:
                setattr(self, hidden_layer, torch.nn.Linear(self.inputs, num_neurons[i], bias=True))
            else:
                setattr(self, hidden_layer, torch.nn.Linear(num_neurons[i - 1], num_neurons[i], bias=True))

        output_layer = "fc" + str(num_hidden_layers + 1)
        setattr(self, output_layer, torch.nn.Linear(num_neurons[size_ - 1], self.outputs, bias=True))

        self.actv = torch.nn.ReLU()

    def forward(self, features, num_neurons):
        size_ = len(num_neurons)
        for i in range(size_):
            layer = "fc" + str(i + 1)
            if i == 0:
                x = getattr(self, layer)(features)
            else:
                x = getattr(self, layer)(x)
            x = self.actv(x)

        output_layer = "fc" + str(size_ + 1)
        output = getattr(self, output_layer)(x)
        return output
