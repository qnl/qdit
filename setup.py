from setuptools import setup, find_packages

setup(
    name="qnlib",
    version="0.1.0",
    packages=find_packages(),
    license="MIT",
    license_files=("LICENSE",),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "numpy",
        "cirq",
        "matplotlib",
        "tqdm",
    ],
    author="Elias Alexander Lehman",
    author_email="eliaslehman@berkeley.edu",
    description="A package to support UC Berkeley's Quantum Nanoelectronics Laboratory",
    python_requires=">=3.7",
)