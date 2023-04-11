from setuptools import setup, find_packages

setup(
    name="morphics",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train = morphics.__main__:main'
        ]
    },
    install_requires=[
        # List your dependencies here
        "torch~=2.0.0",
        "numpy~=1.23.5",
        "torchvision~=0.15.1",
        "astropy~=5.1",
        "astroquery~=0.4.6",
        "pandas~=1.5.3",
        "tqdm~=4.65.0",
    ],
)
