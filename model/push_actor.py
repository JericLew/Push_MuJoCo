import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from model.common.mlp import MLP, ResidualMLP


class BaseActor(nn.Module):
    def __init__(self, action_low=-1, action_high=1):
        super(BaseActor, self).__init__()
        self.action_low = torch.from_numpy(action_low)
        self.action_high = torch.from_numpy(action_high)
    
    def forward(self, obs):
        return
    
    def get_distribution(self, obs):
        key = list(obs.keys())[0]
        device = obs[key].device
        mean, std = self(obs)
        
        scaled_mean = 0.5 * (mean + 1) * (self.action_high - self.action_low) + self.action_low
        dist = Independent(Normal(scaled_mean, std), 1)
        return dist
    
    def to(self, device):
        super(BaseActor, self).to(device)
        self.action_low = self.action_low.to(device)
        self.action_high = self.action_high.to(device)
    
class PushNNActor(BaseActor):
    def __init__(
            self,
            backbone,
            action_low=-1,
            action_high=1,
            state_dim=10,
            action_dim=7,
            mlp_dims=[256, 256, 256, 256],
            activation_type="Mish",
            tanh_output=True,
            residual_style=True,
            use_layernorm=False,
            dropout=0.0,
            fixed_std=None,
            learn_fixed_std=False,
            std_min=0.01,
            std_max=1.0,
            visual_feature_dim=128,
    ):
        super(PushNNActor, self).__init__(action_low=action_low, action_high=action_high)

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
        self.action_dim = action_dim
        input_dim = state_dim + visual_feature_dim
        output_dim = action_dim
        
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        
        ## Base MLP for state and visual feature
        self.mlp_base = model(
            [input_dim] + mlp_dims,
            activation_type=activation_type,
            out_activation_type=activation_type,
            use_layernorm=use_layernorm,
            use_layernorm_final=use_layernorm,
            dropout=dropout,
        )

        ## Mean MLP
        self.mlp_mean = MLP(
            mlp_dims[-1:] + [output_dim],
            out_activation_type="Identity",
        )

        ## Std Output
        if fixed_std is None: # Seperate MLP head for std
            self.mlp_logvar = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
        elif learn_fixed_std: # Learnt Parameter for std
            self.logvar = torch.nn.Parameter(
                torch.log(torch.tensor([fixed_std**2 for _ in range(action_dim)])),
                requires_grad=True,
            )
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(std_max**2)), requires_grad=False
        )
        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std
        self.tanh_output = tanh_output

    def forward(self, obs):
        B = len(obs["state"])
        device = obs["state"].device

        state = obs["state"]
        image = obs["image"]

        ## Handle image
        image_feat = self.backbone(image)
        image_feat = self.compress(image_feat)

        ## MLP
        state = torch.cat((state, image_feat), dim=-1)
        encoded_state = self.mlp_base(state)

        ## Mean Output
        out_mean = self.mlp_mean(encoded_state)
        if self.tanh_output:
            out_mean = torch.tanh(out_mean)

        ## Std Output
        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
        elif self.use_fixed_std:
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(encoded_state)
            out_logvar = torch.tanh(out_logvar)
            out_logvar = self.logvar_min + 0.5 * (self.logvar_max - self.logvar_min) * (
                out_logvar + 1
            ) # put back to full range
            out_scale = torch.exp(0.5 * out_logvar)
        return out_mean, out_scale

class PushNNPrivilegedActor(BaseActor):
    def __init__(
            self,
            action_low=-1,
            action_high=1,
            privileged_dim=9,
            action_dim=2,
            mlp_dims=[256, 256, 256, 256],
            activation_type="Mish",
            tanh_output=True,
            residual_style=True,
            use_layernorm=False,
            dropout=0.0,
            fixed_std=None,
            learn_fixed_std=False,
            std_min=0.01,
            std_max=1.0,
    ):
        super(PushNNPrivilegedActor, self).__init__(action_low=action_low, action_high=action_high)

        ## MLP
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim
        input_dim = privileged_dim
        output_dim = action_dim
        
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        ## Base MLP for state
        self.mlp_base = model(
            [input_dim] + mlp_dims,
            activation_type=activation_type,
            out_activation_type=activation_type,
            use_layernorm=use_layernorm,
            use_layernorm_final=use_layernorm,
            dropout=dropout,
        )

        ## Mean MLP
        self.mlp_mean = MLP(
            mlp_dims[-1:] + [output_dim],
            out_activation_type="Identity",
        )

        ## Std Output
        if fixed_std is None: # Seperate MLP head for std
            self.mlp_logvar = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
        elif learn_fixed_std: # Learnt Parameter for std
            self.logvar = torch.nn.Parameter(
                torch.log(torch.tensor([fixed_std**2 for _ in range(action_dim)])),
                requires_grad=True,
            )
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(std_min**2)), requires_grad=False
        )
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(std_max**2)), requires_grad=False
        )
        self.use_fixed_std = fixed_std is not None
        self.fixed_std = fixed_std
        self.learn_fixed_std = learn_fixed_std
        self.tanh_output = tanh_output

    def forward(self, obs):
        B = len(obs["privileged"])
        device = obs["privileged"].device

        state = obs["privileged"]

        ## MLP
        encoded_state = self.mlp_base(state)

        ## Mean Output
        out_mean = self.mlp_mean(encoded_state)
        if self.tanh_output:
            out_mean = torch.tanh(out_mean)

        ## Std Output
        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
        elif self.use_fixed_std:
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(encoded_state)
            out_logvar = torch.tanh(out_logvar)
            out_logvar = self.logvar_min + 0.5 * (self.logvar_max - self.logvar_min) * (
                out_logvar + 1
            ) # put back to full range
            out_scale = torch.exp(0.5 * out_logvar)
        return out_mean, out_scale