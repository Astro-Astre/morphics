import torch.nn as nn
import torch
import torch.nn.functional as F
def get_output_shape(model, image_dim):
    """Get output shape of a PyTorch model or layer"""
    return model(torch.rand(*(image_dim))).data.shape


class Morphics(nn.Module):
    def __init__(self, model):
        super(Morphics, self).__init__()
        self.model = model
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=9),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
        )
        self.channels = 3
        self.cutout_size = 256
        self.expected_input_shape = (
            1,
            self.channels,
            self.cutout_size,
            self.cutout_size,
        )
        # Calculate the output size of the localization network
        self.ln_out_shape = get_output_shape(
            self.localization, self.expected_input_shape
        )

        # Calculate the input size of the upcoming FC layer
        self.fc_in_size = torch.prod(torch.tensor(self.ln_out_shape[-3:]))

        # Fully-connected regression network predicrs the parameters
        # necessary for an attention tracking transformation
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_in_size, 32), nn.ReLU(), nn.Linear(32, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        ident = [1, 0, 0, 0, 1, 0]
        self.fc_loc[2].bias.data.copy_(torch.tensor(ident, dtype=torch.float))
        self.maxpool = nn.MaxPool2d(4, stride=4)

        self.atf_in_size = 3 * 64 * 64
        self.fc1 = nn.Linear(self.atf_in_size, self.atf_in_size, bias=True)

    def spatial_transform(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_in_size)
        theta = self.fc_loc(xs)
        scale_translation_theta = torch.zeros([theta.shape[0], 6])
        scale_translation_theta[:, 0] = theta[:, 0]  # scale_x
        scale_translation_theta[:, 4] = theta[:, 4]  # scale_y
        scale_translation_theta[:, 2] = theta[:, 2]  # translation_x
        scale_translation_theta[:, 5] = theta[:, 5]  # translation_y

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        x = self.maxpool(x)
        # print(x.shape)
        return x

    def forward(self, x):
        stn = self.spatial_transform(x)
        batch_size = stn.size(0)
        stn = stn.view(batch_size, -1)
        stn = self.fc1(stn)
        stn = stn.view(batch_size, self.channels, 64, 64)
        x = self.model(stn)
        return x, stn
