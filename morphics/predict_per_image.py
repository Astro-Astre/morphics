from utils.utils import *
from models.utils import *

const_bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": 1.0,
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -3.0,
    "type": "Flipout",  # Flipout or Reparameterization
    "moped_enable": False,  # initialize mu/sigma from the dnn weights
    "moped_delta": 0.2,
}
from astropy.io import fits


def load_img(file):
    """
    加载图像，dat和fits均支持，不过仅支持CxHxW
    :param filename: 传入文件名，应当为CHW
    :return: 返回CHW的ndarray
    """
    if ".fits" in file:
        with fits.open(file) as hdul:
            return hdul[0].data.astype(np.float32)
    else:
        raise TypeError


def run():
    model = torch.load("/data/public/renhaoye/morphics/pth/model_148.pt")
    model.train()
    model.cuda()
    sample_mnist = torch.from_numpy(np.array(load_img("/data/public/renhaoye/morphics/dataset/in_decals/agmtn/138"
                                                      ".78548468501958_30.70684135654336.fits")))
    print(type(sample_mnist))
    sample_mnist = sample_mnist.to("cuda:0")
    pred_mnist, epi_mnist_norm, ale_mnist_norm = get_uncertainty_per_image(model, sample_mnist, T=25, normalized=True)
    print(pred_mnist)
    print(epi_mnist_norm)
    print(ale_mnist_norm)



if __name__ == '__main__':
    run()
