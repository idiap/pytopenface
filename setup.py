from setuptools import setup
from setuptools import find_packages

setup(
    name="pytopenface",
    version=open("version.txt", "r").read().strip(),
    description="A PyTorch version of the OpenFace",
    author="Pete Tae-hoon Kim",
    author_email="thnkim@gmail.com",
    maintainer="Olivier Canévet",
    maintainer_email="olivier.canevet@idiap.ch",
    keywords="openface",
    license="Apache 2.0",

    packages=find_packages(),
    zip_safe=False,
    # python_requires=">=3.7",
    install_requires=[ "torch>=1.4.0", "dlib" ],
    package_data={ "pytopenface": ["models/*.pth"] },
    include_package_data=True,
)
