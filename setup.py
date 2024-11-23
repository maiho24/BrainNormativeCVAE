from setuptools import setup, find_packages
import os

def setup_package():
    data = dict(
        name='BrainNormativeCVAE',
        version='0.1.0',
        packages=find_packages(),
        install_requires=[
            'statsmodels==0.13.2',
            'matplotlib==3.5.3',
            'numpy==1.23.2',
            'torch==1.12.1',
            'pandas==1.4.3',
            'scikit-learn==1.3.0',
            'optuna==3.4.0',
            'pyyaml>=6.0.1',
            'tqdm>=4.65.0',
            'setuptools>=64.0.0',
            'openpyxl>=3.1.5'
        ],
        description='A library for running cVAE-based normative model',
        long_description="""
        BrainNormativeCVAE is a library for normative modeling using conditional VAE.
        This implementation extends the normative modelling framework by 
        Lawry Aguila et al. (2022) (https://github.com/alawryaguila/normativecVAE),
        with refinements in both the model architecture and inference approach.
        """,
        author='Mai Ho',
        author_email='mai.ho@unsw.edu.au',
        url='https://github.com/maiho24/BrainNormativeCVAE',
        python_requires='>=3.8',
        entry_points={
            'console_scripts': [
                'brain-cvae-train=scripts.train_model:main',
                'brain-cvae-inference=scripts.run_inference:main',
            ],
        },
        project_urls={
            'Original Implementation': 'https://github.com/alawryaguila/normativecVAE',
        }
    )
    setup(**data)

if __name__ == "__main__":
    setup_package()