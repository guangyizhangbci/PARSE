from setuptools import setup, find_packages

setup(
    name="parse",
    packages=find_packages(),
    version="0.1.0",
    license="MIT",
    description="PARSE - Pytorch",
    author="Guangyi Zhang",
    author_email="guangyi.zhang@utoronto.ca",
    url="https://github.com/guangyizhangbci/PARSE",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "semi-supervised learning",
        "eeg",
        "emotion recognition"
    ],
    install_requires=[
        "einops>=0.3",
        "torch>=1.6",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
