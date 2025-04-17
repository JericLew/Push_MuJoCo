"""
Implementation of Multi-layer Perception (MLP).

Residual model is taken from https://github.com/ALRhub/d3il/blob/main/agents/models/common/mlp.py

Code based on https://github.com/irom-princeton/dppo/blob/main/model/common/mlp.py
"""

import torch
import torch.nn as nn
from collections import OrderedDict

activation_dict = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
    }
)

class MLP(nn.Module):
    def __init__(
            self,
            dim_list,
            append_dim=0,
            append_layers=None,
            activation_type='Tanh',
            out_activation_type='Identity',
            use_layernorm=False,
            use_layernorm_final=False,
            dropout=0.0,
            use_drop_final=False,
            verbose=False,
    ):
        super(MLP, self).__init__()

        self.moduleList = nn.ModuleList()
        self.append_layers = append_layers
        num_layers = len(dim_list) - 1
        for idx in range(num_layers):
            ## Layer Dimensions
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            ## Add Components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layers - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0.0 and (idx < num_layers - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))

            ## Add Activations
            act = (
                activation_dict[activation_type]
                if idx < num_layers - 1
                else activation_dict[out_activation_type]
            )
            layers.append(("act_1", act))

            ## re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            print(f"MLP: {len(self.moduleList)} layers")

    def forward(self, x, append=None):
        for layer_idx, module in enumerate(self.moduleList):
            if append is not None and layer_idx in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = module(x)
        return x
    
class ResidualMLP(nn.Module):
    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        self.layers.append(activation_dict[out_activation_type])

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x

class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "dropout1"):
            x = self.dropout1(x)
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        if hasattr(self, "dropout2"):
            x = self.dropout2(x)
        return x + x_input      
