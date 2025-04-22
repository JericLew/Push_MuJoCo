import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, image_input_shape=(3, 256, 256), feature_dim=512):
        super(ImageEncoder, self).__init__()
        self.image_input_shape = image_input_shape

        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                # nn.BatchNorm2d(out_channels),
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

        if conv_output_dim != feature_dim:
            self.projection = nn.Linear(conv_output_dim, feature_dim)
        else:
            self.projection = nn.Identity()

        self.feature_dim = feature_dim
        # print("Image Encoder Output Shape: ", self.feature_dim)

    def forward(self, image_tensor):
        x = self.encoder(image_tensor)
        return self.projection(x)
    
class DualImageEncoder(nn.Module):
    def __init__(self, image_input_shape=(3, 256, 256), feature_dim=512):
        super(DualImageEncoder, self).__init__()
        self.image_input_shape = image_input_shape
        self.feature_dim = feature_dim

        self.encoder1 = ImageEncoder(image_input_shape, feature_dim)
        self.encoder2 = ImageEncoder(image_input_shape, feature_dim)
        self.compress = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def forward(self, images):
        image1 = images[:, 0, :, :, :]
        image2 = images[:, 1, :, :, :]

        feature1 = self.encoder1(image1)
        feature2 = self.encoder2(image2)

        combined_feature = torch.cat((feature1, feature2), dim=-1)
        compressed_feature = self.compress(combined_feature)

        return compressed_feature