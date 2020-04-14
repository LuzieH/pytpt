import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transitions",
    version="0.0.1",
#    author=" ",
#    author_email=" ",
    description="Implementation of Transition Path Theory for stationary, periodically varying, as well as finite-time Markov chains.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LuzieH/transitions",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy', 'scipy' , 'matplotlib'],
)