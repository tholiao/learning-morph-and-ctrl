import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="learning-morph-and-ctrl",
    version="0.0.1",
    author="Thomas Liao",
    author_email="thomasliao@berkeley.edu",
    description="Package for microrobot simulator experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
)
