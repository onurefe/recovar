# setup.py
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='recovar',  # Replace with your package name
    version='0.1.0',      # Initial release version
    author='Onur Efe',
    author_email='onur.efe44@gmail.com',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='git@github.com:onurefe/recovar.git',
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.14.0',
        'numpy==1.26.0',
        'pandas',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change if using a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10,<3.11',
    include_package_data=True,
    entry_points={
    },
)
