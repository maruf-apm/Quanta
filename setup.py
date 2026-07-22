from setuptools import setup, find_packages

setup(
    name="quanta1",
    version="1.0.0",
    description="Physics-Informed Neural Networks for First-Order ODEs",
    author="Quanta1 Research",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
)
