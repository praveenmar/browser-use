from setuptools import setup, find_packages

setup(
    name="test_framework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "browser-use",
        "langchain",
        "langchain-core",
        "langchain-google-genai",
        "python-dotenv",
        "patchright"
    ],
) 