import setuptools

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import call

with open("README.md","r") as f:
    long_description = f.read()

class Installation(install):
    def run(self):
        call(["pip install -r requirements.txt --no-clean"], shell=True)
        install.run(self)

setuptools.setup(
    name="PyCLUE",
    version="2019.12.05",
    author="Liu Shaoweihua",
    author_email="liushaoweihua@126.com",
    maintainer="CLUE",
    maintainer_email="chineseGLUE@163.com",
    description="Python toolkit for Chinese Language Understanding Evaluation benchmark.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChineseGLUE/PyCLUE",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["tensorflow","requests","numpy"],
    install_requires=["tensorflow","requests","numpy"],
    cmdclass={'install':Installation},
)
