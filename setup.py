import setuptools
from distutils.core import Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

gibbsc = Extension("_gibbs", sources=["gibbs.c"])

setuptools.setup(
    name="dimension",
    version="0.0.2",
    author="Michele Allegra",
    author_email="micheleallegra85@example.com",
    description="Dimension estimation package",
    ext_modules=[gibbsc],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/micheleallegra/hidalgo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
)
