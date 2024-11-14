from setuptools import setup, find_packages

setup(
    name='keratorch',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "torch"
    ],
    # Metadata
    author='Moussa JAMOR',
    author_email='moussajamorsup@gmail.com',
    description="Keratorch: A Keras-style high-level API for building and training models in PyTorch",
    url='https://github.com/JamorMoussa/keratorch',
)