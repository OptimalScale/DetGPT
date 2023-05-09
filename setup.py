import os
from setuptools import find_packages
from setuptools import setup
import subprocess

folder = os.path.dirname(__file__)

__version__ = "0.0.1"


req_path = os.path.join(folder, "requirements.txt")
install_requires = []
if os.path.exists(req_path):
  with open(req_path) as fp:
    install_requires = [line.strip() for line in fp]


readme_path = os.path.join(folder, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
  with open(readme_path, encoding='utf-8') as fp:
    readme_contents = fp.read().strip()

setup(
    name="detgpt",
    version=__version__,
    description="DetGPT.",
    author="Renjie Pi, Jiahui Gao, Shizhe Diao, Rui Pan, Hanze Dong, Jipeng Zhang, Leiwei Yao, Lingpeng Kong, Tong Zhang",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    packages=find_packages("."),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    requires_python=">=3.9",
)