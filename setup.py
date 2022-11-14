

from setuptools import find_packages, setup

setup(
    name="two_peptides",
    version=0.1,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'two_peptides=two_peptides.cli:cli_main',
        ],
    },
    package_data={
        # And include any *.msg files found in the "hello" package, too:
        "two_peptides": ["*.pic"],
    }
)
