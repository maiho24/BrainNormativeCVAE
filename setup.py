from setuptools import setup, find_packages
import os

def setup_package():
    data = dict(
        name='BrainNormativeCVAE',
        version='0.1.0',
        packages=find_packages(),
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
        python_requires='>=3.9',
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
