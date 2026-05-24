from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="amd-bioforge",
    version="0.3.1",
    author="Indie Developer",
    author_email="dev@bioforge.ai",
    description="GPU-Accelerated Protein Folding & Drug Discovery for AMD GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hakuba-pamu/amd-bioforge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
