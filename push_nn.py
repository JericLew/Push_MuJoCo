# continuous action space
# input layer: image (encoded current and target object pos) + 7 joint angles
# output layer: mean and standard deviation for each action

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_output_dim(layers, input_shape):
    """
    Calculate the output dimensions of a tensor after passing through a sequence of layers.
    
    Args:
        layers (nn.Sequential): A PyTorch Sequential block containing the layers.
        input_shape (tuple): The shape of the input tensor (C, H, W).
    
    Returns:
        tuple: The output shape (C, H, W) after passing through the layers.
    """
    # Create a dummy input tensor with the given shape
    dummy_input = torch.randn(1, *input_shape)  # Add batch dimension (1, C, H, W)
    
    # Pass the dummy input through each layer and track the shape
    for layer in layers:
        dummy_input = layer(dummy_input)
    
    # Return the final shape (excluding the batch dimension)
    return dummy_input.shape[1:]


class PushNN(nn.Module):
    def __init__(self, state_dim=10, image_dim=(480, 640, 3), action_dim=7, state_embedding_dim=64):
        """
        PushNN: Neural network for the push task.
        Args:
            state_dim (int): Dimension of the state space. (robot joint angles + end effector position)
            image_dim (tuple): Dimension of the image input. (height, width, channels)
            action_dim (int): Dimension of the action space. (number of actuators)
            embedding_dim (int): Dimension of the embedding space. (size of the hidden state for the neural network)
        """

        super(PushNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.image_dim = image_dim
        self.channel_dim = image_dim[2]
        self.action_dim = action_dim
        self.state_embedding_dim = state_embedding_dim
                
        # for image input -> input_size = [1, 3, 480, 640]; output_size for each layers commented as follows
        self.sequential_block = nn.Sequential(nn.MaxPool2d(2),                                                                              # [1, 3, 240, 320] -> half spatial dimensions
                                              nn.Conv2d(in_channels=self.channel_dim, out_channels=32, kernel_size=3, stride=1, padding=1), # [1, 32, 240, 320]
                                              nn.ReLU(),
                                              nn.MaxPool2d(2),                                                                              # [1, 32, 120, 160] -> half spatial dimensions
                                              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),               # [1, 64, 120, 160]
                                              nn.ReLU(),
                                              nn.MaxPool2d(2),                                                                              # [1, 64, 60, 80] -> half spatial dimensions
                                              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),              # [1, 128, 60, 80]
                                              nn.ReLU(),
                                              nn.MaxPool2d(2),                                                                              # [1, 128, 30, 40] -> half spatial dimensions
                                              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),              # [1, 64, 30, 40]
                                              nn.ReLU(),
                                              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),               # [1, 32, 30, 40]
                                              nn.ReLU(),
                                              nn.MaxPool2d(2),                                                                              # [1, 32, 15, 20] -> half spatial dimensions
                                              nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),               # [1, 16, 15, 20]
                                              nn.ReLU(),
                                              nn.MaxPool2d(2),                                                                              # [1, 16, 8, 20] -> half spatial dimensions
                                              nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),                # [1, 8, 7, 10]
                                              nn.ReLU(),
                                              nn.Flatten() # [1, 560] -> flatten into a single vector of 8 x 7 x 10
                                              )
        
        image_input_shape = (self.channel_dim, image_dim[0], image_dim[1])  #  (Channels, Height, Width)
        self.image_encoder_output_dim = calculate_output_dim(self.sequential_block, image_input_shape)[0] # flattened
        print("Image Encoder Output Shape: ", self.image_encoder_output_dim)
        
        # for joint angles input
        self.fully_connected_1 = nn.Sequential(nn.Linear(state_dim, state_embedding_dim),
                                               nn.ReLU())
        
        # for skip connection after concat
        self.combined_embedding_dim = self.image_encoder_output_dim + state_embedding_dim
        self.fully_connected_2 = nn.Sequential(nn.Linear(self.combined_embedding_dim, self.combined_embedding_dim), 
                                               nn.ReLU())
        self.fully_connected_3 = nn.Linear(self.combined_embedding_dim, self.combined_embedding_dim)

        # TODO: why this feels so sus... TT
        # Policy layers: mean and log(std) for each action
        self.policy_mean = nn.Linear(self.combined_embedding_dim, self.action_dim)
        self.policy_logstd = nn.Linear(self.combined_embedding_dim, self.action_dim)
        
        # Value layer
        self.value_layer = nn.Linear(self.combined_embedding_dim, 1)

        self.to(self.device)
        

    def forward(self, obs): 
        # obs = {"state": state, "image": image}
        image_tensor = obs['image'] # shape: (B, 480, 640, 3), dtype: uint8, type: <class 'numpy.ndarray'> (B, H, W, C)
        image_tensor = image_tensor.permute(0, 3, 1, 2) # change shape from (B, H, W, C) to (B, C, H, W)
        # TODO: do we need preprocessing? resize? normalize??? (mean of 0 and a standard deviation of 1)
        # image_tensor = image_tensor.astype(np.float32) / 255.0 # normalize to [0, 1]
        flat = self.sequential_block(image_tensor) # [1, 560]

        state_tensor = obs['state'] # shape: (B, 10), dtype: float32, type: <class 'numpy.ndarray'> (B, joint angles + end effector position)
        qpos_layer = self.fully_connected_1(state_tensor) # [B, state_embedding_dim]

        hidden_input = torch.concat([flat, qpos_layer], 1) # [B, state_embedding_dim + image_encoder_output_dim]
        h1 = self.fully_connected_2(hidden_input)
        h2 = self.fully_connected_3(h1)
        h3 = F.relu(h2 + hidden_input) # [B,  state_embedding_dim + image_encoder_output_dim]
        
        # TODO: skipped LSTM for now

        # Policy head
        mean = self.policy_mean(h3)
        log_std = self.policy_logstd(h3)
        std = torch.exp(log_std)  # Ensure positive standard deviation

        # Value head
        value = self.value_layer(h3)

        return (mean, std), value
