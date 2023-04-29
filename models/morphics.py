import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def get_output_shape(model, image_dim):
    """Get output shape of a PyTorch model or layer"""
    return model(torch.rand(*(image_dim))).data.shape


# 将Linear输出转换为12个点的集合
def convert_to_points(linear_output):
    points = []
    for b in range(linear_output.size(0)):
        batch_points = []
        for i in range(0, linear_output.size(1), linear_output.size(1) // 6):
            batch_points.append(torch.tensor([0, 0], dtype=torch.float32, device=linear_output.device))
            batch_points += [linear_output[b, i:i + 2], linear_output[b, i + 2:i + 4], linear_output[b, i + 4:i + 6],
                             linear_output[b, i + 6:i + 8], linear_output[b, i + 8:i + 10]]
            batch_points.append(torch.tensor([1, 1], dtype=torch.float32, device=linear_output.device))
        points.append(torch.stack(batch_points))
    return torch.stack(points)


# 定义非线性拟合函数
def nonlinear_function(x, points, degree=3):
    points = points.cpu().detach().numpy()
    coeff_x = np.polyfit(points[:, 0], points[:, 1], degree)
    f_x = np.polyval(coeff_x, x.cpu().numpy())
    f_x = np.clip(f_x, 0, 1)
    return torch.tensor(f_x, dtype=torch.float32, device=x.device)


def apply_function_to_image(linear_output, input_image):
    batch_size, channels, height, width = input_image.size()
    points = convert_to_points(linear_output)

    x = torch.linspace(0, 1, width).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(input_image.device)
    output_image = torch.zeros_like(input_image)
    for b in range(batch_size):
        for c in range(channels):
            group_points = points[b][c * 2:(c * 2) + 12]
            y = nonlinear_function(x, group_points).unsqueeze(1)
            output_image[b, c] = input_image[b, c] * y
    return output_image


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
        self.ln_out_shape = get_output_shape(self.localization, self.expected_input_shape)
        self.fc_in_size = torch.prod(torch.tensor(self.ln_out_shape[-3:]))
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_in_size, 32), nn.ReLU(), nn.Linear(32, 6)
        )
        self.atf = nn.Sequential(
            nn.Linear(self.fc_in_size, 32, bias=True), nn.ReLU(), nn.Linear(32, 60, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.uniform_(self.atf[-1].weight, a=0, b=1)
        nn.init.uniform_(self.atf[-1].bias, a=0, b=1)
        self.fc_loc[2].weight.data.zero_()
        ident = [1, 0, 0, 0, 1, 0]
        self.fc_loc[2].bias.data.copy_(torch.tensor(ident, dtype=torch.float))

        self.maxpool = nn.AvgPool2d(4, stride=4)

    def spatial_transform(self, x):
        # device = x.device
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_in_size)
        theta = self.fc_loc(xs)
        # scale_translation_theta = torch.zeros([theta.shape[0], 6])
        # scale_translation_theta[:, 0] = theta[:, 0]  # scale_x
        # scale_translation_theta[:, 4] = theta[:, 4]  # scale_y
        # scale_translation_theta[:, 2] = theta[:, 2]  # translation_x
        # scale_translation_theta[:, 5] = theta[:, 5]  # translation_y

        # theta = scale_translation_theta.view(-1, 2, 3).to(device)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def auto_stretch(self, x):
        x = self.localization(x)
        x = x.view(-1, self.fc_in_size)
        atf = self.atf(x)
        # atf = self.sigmoid(atf)
        atf = atf.view(-1, 60)
        return atf

    def forward(self, x):
        stn = self.spatial_transform(x)
        # atf = self.auto_stretch(stn)
        # stned = apply_function_to_image(atf, stn)
        stn = self.maxpool(stn)
        x = self.model(stn)
        return x, stn
