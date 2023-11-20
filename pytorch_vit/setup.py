from setuptools import setup, find_packages

setup(
    name='pytorch_vit', 
    version='0.1.0',  # Package's version
    author='Preston Govender',
    author_email='prestongov@gmail.com', 
    description='Vision transformers for small datasets.',  
    long_description=open('README.md').read(),  # Long description read from the the readme file
    long_description_content_type='text/markdown',  # This is important to ensure that README.md is rendered correctly
    url='https://github.com/pressi-g/pytorch-vit',  # Link to repository
    packages=find_packages(),  # Finds all python modules (folders with __init__.py) automatically
    install_requires=[
        'numpy',  # List all the dependencies that your package needs
        'torch',
        # Add other dependencies as necessary
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',  # Minimum version requirement of the package
    # Add additional keywords relevant to your package
    keywords='vision transformer, neural networks, image classification',
)
