from setuptools import setup, find_packages

setup(
    name="boltzmann_simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Boltzmann collision simulation package",
    python_requires=">=3.6",
) 