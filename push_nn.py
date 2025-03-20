# continuous action space
# input layer: image (encoded current and target object pos) + 7 joint angles
# output layer: mean and standard deviation for each action

import torch
import torch.nn as nn
import torch.nn.functional as F

# parameters for training
# RNN_SIZE = 128      # size of the hidden state for the recurrent neural network (LSTM) 
# GOAL_REPR_SIZE = 12 # size of the goal representation (number of features for goal representation) in the neural network 


class PushNN(nn.Module):
    def __init__(self, action_size=7, qpos_repr_size=64):
        super(PushNN, self).__init__()
        self.NUM_CHANNEL = 3 # RGB image
        self.IMAGE_HIDDEN_SIZE = 560 # TODO: make this into a parameter??
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # action_size = num of joint angles
                
        # for image input -> input_size = [1, 3, 480, 640]; output_size for each layers commented as follows
        self.sequential_block = nn.Sequential(nn.MaxPool2d(2),                                                                              # [1, 3, 240, 320] -> half spatial dimensions
                                              nn.Conv2d(in_channels=self.NUM_CHANNEL, out_channels=32, kernel_size=3, stride=1, padding=1), # [1, 32, 240, 320]
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
        
        # for joint angles input
        self.fully_connected_1 = nn.Sequential(nn.Linear(action_size, qpos_repr_size), # input size = [7], output size = [64]
                                               nn.ReLU())
        
        # for skip connection after concat
        self.actual_hidden_size = self.IMAGE_HIDDEN_SIZE + qpos_repr_size
        self.fully_connected_2 = nn.Sequential(nn.Linear(self.actual_hidden_size, self.actual_hidden_size), 
                                               nn.ReLU())
        self.fully_connected_3 = nn.Linear(self.actual_hidden_size, self.actual_hidden_size)

        # TODO: why this feels so sus... TT
        # Policy layers: mean and log(std) for each action
        self.policy_mean = nn.Linear(self.actual_hidden_size, action_size)
        self.policy_logstd = nn.Linear(self.actual_hidden_size, action_size)
        
        # Value layer
        self.value_layer = nn.Linear(self.actual_hidden_size, 1)

        self.to(self.device)
        

    def forward(self, obs): 
        # obs = {"state": state, "image": image}
        image = obs['image'] # shape: (480, 640, 3), dtype: uint8, type: <class 'numpy.ndarray'>
        
        image_tensor = torch.from_numpy(image).to(self.device).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)   # .permute() change shape from (H, W, C) → (C, H, W); 
                                                                    # unsqueeze.(0) add a batch dimension, final shape (1, C, H, W)
        # TODO: do we need preprocessing? resize? normalize??? (mean of 0 and a standard deviation of 1)
        flat = self.sequential_block(image_tensor) # [1, 560]
        # print("image_tensor: ", image_tensor.shape)
        # print("flat: ", flat.shape)

        qpos = obs['state'][:7] # state = np.concatenate([robot_joint_angles, end_effector_pos])
        qpos_tensor = torch.from_numpy(qpos).to(self.device).float().unsqueeze(0) # [1, 7]
        qpos_layer = self.fully_connected_1(qpos_tensor) # [1, 64]
        # print("qpos_tensor: ", qpos_tensor.shape)
        # print("qpos_layer: ", qpos_layer.shape)

        hidden_input = torch.concat([flat, qpos_layer], 1) # [1, 624]
        h1 = self.fully_connected_2(hidden_input)
        h2 = self.fully_connected_3(h1)
        h3 = F.relu(h2 + hidden_input) # [1, 624] --> output (TODO: skipped LSTM for now)
        # print("h3: ", h3.shape)

        # Policy head
        mean = self.policy_mean(h3)
        log_std = self.policy_logstd(h3)
        std = torch.exp(log_std)  # Ensure positive standard deviation
        print("NN_mean: ", mean.shape, mean)
        print("NN_log_std: ", log_std.shape, log_std)

        # Value head
        value = self.value_layer(h3)
        print("NN_value: ", value.shape, value)

        return (mean, std), value
