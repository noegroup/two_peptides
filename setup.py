

from setuptools import find_packages, setup

setup(
    name="two_peptides",
    version=0.1,
    packages=find_packages(),
    entrypoints={
        'console_scripts': [
            'two_peptides=two_peptides.run:main',
        ],
    }
)