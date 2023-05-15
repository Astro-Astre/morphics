import torch.nn as nn
import torch
import torch.nn.functional as F
from astropy.io import fits


def get_output_shape(model, image_dim):
    """Get output shape of a PyTorch model or layer"""
    return model(torch.rand(*(image_dim))).data.shape


def rescale_image(input_tensor: torch.Tensor, shadow_clip, highlight_clip) -> torch.Tensor:
    device = input_tensor.device

    # 线性缩放到 [shadow_clip, highlight_clip] 区间
    rescaled_tensor = (input_tensor - shadow_clip) / (highlight_clip - shadow_clip)

    # 将低于阴影裁剪点的像素值设置为 0（黑色），高于高光裁剪点的像素值设置为 1（白色）
    rescaled_tensor = torch.where(input_tensor < shadow_clip, torch.tensor(0.0).to(device), rescaled_tensor)
    rescaled_tensor = torch.where(input_tensor > highlight_clip, torch.tensor(1.0).to(device), rescaled_tensor)

    return rescaled_tensor

def mtf(img, mtf_m):
    assert img.shape == (img.shape[0], img.shape[1], img.shape[2], img.shape[3]), "Input must have shape (B, C, H, W)"
    assert mtf_m.shape == (img.shape[0], 6), "MTF matrix must have shape (B, 6)"

    black, midtone = mtf_m[:, :3], mtf_m[:, 3:]
    white = torch.ones_like(black)
    if isinstance(img, torch.Tensor):
        assert torch.all(black < white), "Threshold values must satisfy black < midtone < white for each channel"
        device = img.device
        black, midtone, white = black.to(device).unsqueeze(2).unsqueeze(3), midtone.to(device).unsqueeze(2).unsqueeze(
            3), white.to(device).unsqueeze(2).unsqueeze(3)
        result = rescale_image(img, black, white)
        result = (midtone - 1) * result / ((2 * midtone - 1) * result - midtone)

    else:
        raise ValueError("Input must be either NumPy array or PyTorch tensor")

    return result


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
        # self.localization1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11),
        #     nn.MaxPool2d(3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 96, kernel_size=9),
        #     nn.MaxPool2d(3, stride=2),
        #     nn.ReLU(),
        # )
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
        # self.atf = nn.Sequential(
        #     nn.Linear(self.fc_in_size, 32, bias=True), nn.ReLU(), nn.Linear(32, 6, bias=True),
        # )
        # nn.init.normal_(self.atf[-1].weight, mean=0.1, std=0.1)
        # nn.init.normal_(self.atf[-1].bias, mean=0.1, std=0.1)
        self.fc_loc[2].weight.data.zero_()
        ident = [1, 0, 0, 0, 1, 0]
        self.fc_loc[2].bias.data.copy_(torch.tensor(ident, dtype=torch.float))

    def spatial_transform(self, x):
        # device = x.device
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_in_size)
        theta = self.fc_loc(xs)
        # scale_translation_theta = torch.zeros([theta.shape[0], 6])
        # scale_translation_theta[:, 0] = theta[:, 0]  # scale_x
        # scale_translation_theta[:, 4] = theta[:, 4]  # scale_y
        # scale_translation_theta[:, 2] = theta[:, 1]  # translation_x
        # scale_translation_theta[:, 5] = theta[:, 3]  # translation_y

        # theta = scale_translation_theta.view(-1, 2, 3).to(device)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        # x = self.maxpool(x)
        return x

    # def auto_stretch(self, x):
    #     x = self.localization1(x)
    #     x = x.view(-1, self.fc_in_size)
    #     atf = self.atf(x)
    #     # atf = self.sigmoid(atf)
    #     black = 0.1 * torch.sigmoid(atf[:, :3])
    #     midtone = black + 0.4 * torch.sigmoid(atf[:, 3:6])
    #     out = torch.cat([black, midtone], dim=1)
    #     return out

    def forward(self, x):
        # atf = self.auto_stretch(x)
        # atfed = mtf(x, atf)
        stn = self.spatial_transform(x)
        # fits.writeto(f'/data/public/renhaoye/2.fits', stn[0].cpu().detach().numpy(), overwrite=True)
        x = self.model(stn)
        return x, stn
