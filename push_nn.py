'''
NOT USED ANYMORE REFER to model/*
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            activation_type='Mish',
            out_activation_type='Identity',
            use_layernorm=False,
            use_layernorm_final=False,
            dropout_p=0.0,
            dropout_final=False,
            verbose=False,
    ):
        super(MLP, self).__init__()
        self.module_list = nn.ModuleList()
        self.append_layers = append_layers
        num_layers = len(dim_list) - 1
        for idx in range(num_layers):
            ## Layer Dimensions
            in_dim = dim_list[idx]
            out_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                in_dim += append_dim
            linear_layer = nn.Linear(in_dim, out_dim)

            ## Add Components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layers - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(out_dim)))
            if dropout_p > 0.0 and (idx < num_layers - 1 or dropout_final):
                layers.append(("dropout_1", nn.Dropout(dropout_p)))

            ## Add Activations
            act = (
                activation_dict[activation_type]
                if idx < num_layers - 1
                else activation_dict[out_activation_type]
            )
            layers.append(("act_1", act))

            ## re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.module_list.append(module)
        if verbose:
            print(f"MLP: {len(self.module_list)} layers")

    def forward(self, x, append=None):
        for layer_idx, module in enumerate(self.module_list):
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
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input      

class ImageEncoder(nn.Module):
    def __init__(self, image_input_shape=(3, 256, 256), embedding_dim=512):
        super(ImageEncoder, self).__init__()
        self.image_input_shape = image_input_shape

        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            conv_block(3, 32),    # (B, 32, 128, 128)
            conv_block(32, 64),   # (B, 64, 64, 64)
            conv_block(64, 128),  # (B, 128, 32, 32)
            conv_block(128, 128), # (B, 128, 16, 16)
            conv_block(128, 64),  # (B, 64, 8, 8)
            conv_block(64, 32),   # (B, 32, 4, 4)
            nn.Flatten(),         # (B, 512)
        )

        dummy_input = torch.zeros(1, *image_input_shape)
        conv_output_dim = self.encoder(dummy_input).shape[1]

        if conv_output_dim != embedding_dim:
            self.projection = nn.Linear(conv_output_dim, embedding_dim)
        else:
            self.projection = nn.Identity()

        self.embedding_dim = embedding_dim
        print("Image Encoder Output Shape: ", self.embedding_dim)

    def forward(self, image_tensor):
        x = self.encoder(image_tensor)
        return self.projection(x)

class StateEncoder(nn.Module):
    def __init__(self, state_dim=14, embedding_dim=64, hidden_dim=128, dropout_p=0.1):
        """
        StateEncoder: Neural network for the state space.
        Args:
            state_dim (int): Input dimension (joint angles + ee pos + ee quat).
            embedding_dim (int): Output embedding size.
            hidden_dim (int): Size of hidden layers.
            dropout_p (float): Dropout probability.
        """
        super(StateEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # or nn.BatchNorm1d(hidden_dim)
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )

    def forward(self, state_tensor):
        """
        Args:
            state_tensor (torch.Tensor): (B, state_dim)
        Returns:
            torch.Tensor: (B, embedding_dim)
        """
        return self.net(state_tensor)

class PushNN(nn.Module):
    def __init__(self, state_dim=10, image_dim=(3, 256, 256), action_dim=7, state_embedding_dim=64, image_embedding_dim=128):
        """
        PushNN: Neural network for the push task.
        Args:
            state_dim (int): Dimension of the state space. (joint angles + ee pos + ee quat)
            image_dim (tuple): Dimension of the image input. (C, H, W)
            action_dim (int): Dimension of the action space. (number of actuators)
            state_embedding_dim (int): Dimension of the state embedding.
            image_embedding_dim (int): Dimension of the image embedding.
        """
        super(PushNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.image_dim = image_dim
        self.channel_dim = image_dim[2]
        self.action_dim = action_dim
        self.state_embedding_dim = state_embedding_dim
                
        # Image encoder: top and side images
        self.top_image_encoder = ImageEncoder(image_input_shape=image_dim, embedding_dim=image_embedding_dim)
        self.side_image_encoder = ImageEncoder(image_input_shape=image_dim, embedding_dim=image_embedding_dim)
        self.combined_image_embedding_dim = self.top_image_encoder.embedding_dim + self.side_image_encoder.embedding_dim
        
        # State encoder
        self.state_encoder = StateEncoder(state_dim=state_dim, embedding_dim=state_embedding_dim, hidden_dim=128, dropout_p=0.1)

        # Fusion
        self.combined_embedding_dim = self.combined_image_embedding_dim + state_embedding_dim
        self.fully_connected_2 = nn.Sequential(
            nn.Linear(self.combined_embedding_dim, self.combined_embedding_dim), 
            nn.ReLU()
            )
        self.fully_connected_3 = nn.Linear(self.combined_embedding_dim, self.combined_embedding_dim)

        # Policy head
        self.policy_mean = nn.Sequential(
            nn.Linear(self.combined_embedding_dim, self.combined_embedding_dim),
            nn.Tanh(),
            nn.Linear(self.combined_embedding_dim, action_dim),
            nn.Tanh(),
        )
        # self.policy_mean = nn.Linear(self.combined_embedding_dim, self.action_dim)

        # Learnable global log_std
        self.policy_logstd = nn.Sequential(
            nn.Linear(self.combined_embedding_dim, self.combined_embedding_dim),
            nn.Tanh(),
            nn.Linear(self.combined_embedding_dim, action_dim),
        )
        # self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.0))  # Learnable tensor, initialized to -1
        
        # Value head
        self.value_layer = nn.Linear(self.combined_embedding_dim, 1)

        self.to(self.device)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        super(PushNN, self).eval()
        # self.log_std = nn.Parameter(torch.full((self.action_dim,), -3.0))  # Reset log_std to 0 for evaluation
        

    def forward(self, obs):
        """
        Forward pass through the neural network.
        Args:
            obs (dict): Dictionary containing the state and image observations.
                - "state": (B, state_dim) tensor of joint angles + end effector position, float32
                - "image": (B, N, C, H, W) tensor of the image, float32 0-1 RGB, N[0, 1] = top, side
        Returns:
            tuple: (mean, std), value
                - mean: (B, action_dim) tensor of the mean for each action
                - std: (B, action_dim) tensor of the standard deviation for each action
                - value: (B, 1) tensor of the value function
        """
        image_tensor = obs['image']
        top_image_tensor = image_tensor[:, 0, :, :, :] # [B, 3, 256, 256]
        side_image_tensor = image_tensor[:, 1, :, :, :] # [B, 3, 256, 256]
        top_image_feature = self.top_image_encoder(top_image_tensor) # [B, image_embedding_dim]
        side_image_feature = self.side_image_encoder(side_image_tensor) # [B, image_embedding_dim]
        combined_image_feature = torch.concat([top_image_feature, side_image_feature], 1) # [B, 2 * image_embedding_dim]

        state_tensor = obs['state'] # shape: (B, 10), dtype: float32, type: <class 'numpy.ndarray'> (B, joint angles + end effector position)
        state_feature = self.state_encoder(state_tensor) # [B, state_embedding_dim]

        hidden_input = torch.concat([combined_image_feature, state_feature], 1) # [B, state_embedding_dim + 2 * image_embedding_dim]
        h1 = self.fully_connected_2(hidden_input)
        h2 = self.fully_connected_3(h1)
        h3 = F.relu(h2 + hidden_input) # [B, state_embedding_dim + 2 * image_embedding_dim]
        
        # TODO: skipped LSTM for now

        # Policy head
        mean = self.policy_mean(h3) # -1.0 to 1.0 (tanh)
        # std = torch.exp(self.log_std).expand_as(mean)  # Broadcasted per action
        log_std = self.policy_logstd(h3)
        log_std = torch.clamp(log_std, -4, 1)  # reasonable log std range
        std = torch.exp(log_std)  # Ensure positive standard deviation

        # Value head
        value = self.value_layer(h3)

        return (mean, std), value

class PushNNPrivileged(nn.Module):
    def __init__(self, privileged_dim=9, action_dim=2, state_embedding_dim=64):
        """
        PushNNPrivileged: Neural network for the push task. With privileged information.
        Args:
            privileged_dim (int): Dimension of the privileged state space. (ee pos + object pos + target pos)
            action_dim (int): Dimension of the action space. (delta ee x, delta ee y)
            state_embedding_dim (int): Dimension of the state embedding.
        """
        super(PushNNPrivileged, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim
        self.state_embedding_dim = state_embedding_dim
        self.hidden_dim = 2 * state_embedding_dim
                        
        # Privileged State encoder
        self.privileged_encoder = StateEncoder(state_dim=privileged_dim, embedding_dim=state_embedding_dim, hidden_dim=self.hidden_dim, dropout_p=0.1)
        self.privileged_encoder2 = StateEncoder(state_dim=state_embedding_dim, embedding_dim=state_embedding_dim, hidden_dim=self.hidden_dim, dropout_p=0.1)
        self.privileged_encoder3 = StateEncoder(state_dim=state_embedding_dim, embedding_dim=state_embedding_dim, hidden_dim=self.hidden_dim, dropout_p=0.1)

        # Policy head
        self.policy_mean = nn.Sequential(
            StateEncoder(state_dim=state_embedding_dim, embedding_dim=state_embedding_dim, hidden_dim=self.hidden_dim, dropout_p=0.1),
            nn.Linear(self.state_embedding_dim, self.state_embedding_dim),
            nn.Tanh(),
            nn.Linear(self.state_embedding_dim, action_dim),
            nn.Tanh(),
        )
        # self.policy_mean = nn.Linear(self.combined_embedding_dim, self.action_dim)

        # Learnable global log_std
        self.policy_std = nn.Sequential(
            StateEncoder(state_dim=state_embedding_dim, embedding_dim=state_embedding_dim, hidden_dim=self.hidden_dim, dropout_p=0.1),
            nn.Linear(self.state_embedding_dim, self.state_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.state_embedding_dim, action_dim),
            nn.Softplus(),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.0))  # Learnable tensor, initialized to -1
        
        # Value head
        self.value_layer = nn.Linear(self.state_embedding_dim, 1)

        self.to(self.device)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        super(PushNN, self).eval()
        # self.log_std = nn.Parameter(torch.full((self.action_dim,), -3.0))  # Reset log_std to 0 for evaluation
        

    def forward(self, obs):
        privileged_tensor = obs['privileged'] # shape: (B, 10), dtype: float32, type: <class 'numpy.ndarray'> (B, object pos + object quat + target pos)
        privileged_feature = self.privileged_encoder(privileged_tensor) # [B, state_embedding_dim]
        h1 = self.privileged_encoder2(privileged_feature)
        h1 = F.relu(h1 + privileged_feature) # [B, state_embedding_dim]
        h2 = self.privileged_encoder3(h1)
        h2 = F.relu(h2 + h1)

        # TODO: skipped LSTM for now

        # Policy head
        mean = self.policy_mean(h2) # -1.0 to 1.0 (tanh)
        # std = self.policy_std(h2) # 0 to inf (softplus)
        log_std = torch.clamp(self.log_std, -4, 1)  # reasonable log std range
        log_std = log_std.expand_as(mean)
        std = torch.exp(log_std)  # Ensure positive standard deviation

        # Value head
        value = self.value_layer(h2)

        return (mean, std), value
    
class PushNNPrivilegedActor(nn.Module):
    def __init__(
            self,
            privileged_dim=9,
            action_dim=2,
            mlp_dims=[256, 256, 256],
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
        super(PushNNPrivilegedActor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim
        input_dim = privileged_dim
        output_dim = action_dim
        
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        if fixed_std is None:
            # learning std
            self.mlp_base = model(
                [input_dim] + mlp_dims,
                activation_type=activation_type,
                out_activation_type=activation_type,
                use_layernorm=use_layernorm,
                use_layernorm_final=use_layernorm,
                dropout=dropout,
            )
            self.mlp_mean = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
            self.mlp_logvar = MLP(
                mlp_dims[-1:] + [output_dim],
                out_activation_type="Identity",
            )
        else:
            # no separate head for mean and std
            self.mlp_mean = model(
                [input_dim] + mlp_dims + [output_dim],
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
                dropout=dropout,
            )
            if learn_fixed_std:
                # initialize to fixed_std
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

        state = obs["privileged"]  # shape: (B, 10), dtype: float32, type: <class 'numpy.ndarray'> (B, object pos + object quat + target pos)

        # mlp
        if hasattr(self, "mlp_base"):
            state = self.mlp_base(state)
        out_mean = self.mlp_mean(state)
        if self.tanh_output:
            out_mean = torch.tanh(out_mean)

        if self.learn_fixed_std:
            out_logvar = torch.clamp(self.logvar, self.logvar_min, self.logvar_max)
            out_scale = torch.exp(0.5 * out_logvar)
        elif self.use_fixed_std:
            out_scale = torch.ones_like(out_mean).to(device) * self.fixed_std
        else:
            out_logvar = self.mlp_logvar(state)
            out_logvar = torch.tanh(out_logvar)
            out_logvar = self.logvar_min + 0.5 * (self.logvar_max - self.logvar_min) * (
                out_logvar + 1
            )  # put back to full range
            out_scale = torch.exp(0.5 * out_logvar)
        return out_mean, out_scale
    
class PushNNPrivilegedCritic(nn.Module):
    def __init__(
            self,
            privileged_dim=9,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=False,
            residual_style=True,
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
        )

    def forward(self, obs):
        state = obs["privileged"]  # shape: (B, 10), dtype: float32, type: <class 'numpy.ndarray'> (B, object pos + object quat + target pos)
        value = self.critic(state)
        return value