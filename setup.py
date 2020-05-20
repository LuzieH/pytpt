import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytpt",
    version="0.0.1",
    author="Luzie Helfmann and Enric Ribera Borrell",
    author_email="luzie.helfmann@fu-berlin.de",
    description="Implementation of Transition Path Theory for stationary, periodically varying, and finite-time Markov chains.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LuzieH/pytpt",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
    ],
)
