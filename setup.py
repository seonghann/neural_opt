"""
Install package
>>> python setup.py develop
"""

import setuptools

setuptools.setup(
    name="NeuralOpt",
    version="0.0.1",
    author="Jeheon Woo",
    author_email="woojh@kaist.ac.kr",
    description="Riemannian Denoising Score Matching",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9.16",
)
