import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="leak-detect",
    version="0.0.1",
    author="Abhay Pawar",
    author_email="abhayspawar@gmail.com",
    description="Detect leakage in ML datasets using complex numbers and NANs",
    long_description=(
        "Leak-detect helps with detecting data leakages "
        "in your data creation pipeline using complex numbers and NANs. "
        "It enables you to test for data leakage with ease and high accuracy, "
        "while treating your data creation pipeline as a black-box."
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/abhayspawar/leak-detect",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas", "numpy"],
)
