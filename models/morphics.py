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
            nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=9),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(True),
        )
        self.expected_input_shape = (
            1,
            3,
            256,
            256,
        )
        self.ln_out_shape = get_output_shape(
            self.localization, self.expected_input_shape
        )
        # Calculate the input size of the upcoming FC layer
        self.fc_in_size = torch.prod(torch.tensor(self.ln_out_shape[-3:]))
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_in_size, 32), nn.ReLU(True), nn.Linear(32, 3)
        )

    def spatial_transform(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_in_size)
        crop_para = self.fc_loc(xs)

        # Initialize the transformation matrix
        theta = torch.zeros(crop_para.shape[0], 2, 3).to(crop_para.device)

        theta[:, 0, 0] = crop_para[:, 0]
        theta[:, 1, 1] = crop_para[:, 0]
        theta[:, [0, 1], [2, 2]] = crop_para[:, 1:]

        # size = [x.size(0), x.size(1), 128, 128]
        # grid = F.affine_grid(theta, size, align_corners=True)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward(self, x):
        x = self.spatial_transform(x)
        x = self.model(x)

        return x
