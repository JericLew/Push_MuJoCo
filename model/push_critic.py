import torch
import torch.nn as nn

from model.common.mlp import MLP, ResidualMLP


class PushNNCritic(nn.Module):
    def __init__(
            self,
            backbone,
            state_dim=10,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=False,
            residual_style=True,
            dropout=0.0,
            visual_feature_dim=128,
    ):
        super(PushNNCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Vision Backbone
        self.backbone = backbone
        self.compress = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, visual_feature_dim),
            nn.LayerNorm(visual_feature_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        ## MLP
        self.state_dim = state_dim
        input_dim = state_dim + visual_feature_dim
        output_dim = 1
        
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.critic = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, obs):
        state = obs["state"]
        image = obs["image"]

        ## Handle image
        image_feat = self.backbone(image)
        image_feat = self.compress(image_feat)

        state = torch.cat((state, image_feat), dim=-1)
        value = self.critic(state)
        return value

class PushNNPrivilegedCritic(nn.Module):
    def __init__(
            self,
            privileged_dim=9,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=False,
            residual_style=True,
            dropout=0.0,
    ):
        super(PushNNPrivilegedCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.privileged_dim = privileged_dim
        input_dim = privileged_dim
        output_dim = 1
        
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.critic = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
            dropout=dropout,
        )

    def forward(self, obs):
        state = obs["privileged"]
        value = self.critic(state)
        return value