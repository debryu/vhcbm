from setuptools import setup, find_packages
        
setup(
    name='argus',
    version='0.1',
    description='',
    author='itsme',
    author_email='itsme@myaffiliation',
    packages=find_packages(),
    install_requires=[
        "loguru",
        "matplotlib",
        "wandb",
        "seaborn",
        "gpytorch",
        "CQA @ git+http://github.com/debryu/CQA.git"
        ],  
)