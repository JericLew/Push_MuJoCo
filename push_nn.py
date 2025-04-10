# continuous action space
# input layer: image (encoded current and target object pos) + 7 joint angles
# output layer: mean and standard deviation for each action

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, state_dim=10, embedding_dim=64, hidden_dim=128, dropout_p=0.1):
        """
        StateEncoder: Neural network for the state space.
        Args:
            state_dim (int): Input dimension (joint angles + end-effector position).
            embedding_dim (int): Output embedding size.
            hidden_dim (int): Size of hidden layers.
            dropout_p (float): Dropout probability.
        """
        super(StateEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # or nn.BatchNorm1d(hidden_dim)
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
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
            state_dim (int): Dimension of the state space. (robot joint angles + end effector position)
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
        self.policy_mean = nn.Linear(self.combined_embedding_dim, self.action_dim)

        # Learnable global log_std
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))  # Learnable tensor, not from network
        # self.policy_logstd = nn.Linear(self.combined_embedding_dim, self.action_dim)
        
        # Value head
        self.value_layer = nn.Linear(self.combined_embedding_dim, 1)

        self.to(self.device)
        

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
        mean = self.policy_mean(h3)
        std = torch.exp(self.log_std).expand_as(mean)  # Broadcasted per action
        # log_std = self.policy_logstd(h3)
        # log_std = torch.clamp(log_std, -20, 2)  # reasonable log std range
        # std = torch.exp(log_std)  # Ensure positive standard deviation

        # Value head
        value = self.value_layer(h3)

        return (mean, std), value
