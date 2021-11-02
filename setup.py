import setuptools

from pathlib import Path

long_description = Path('README.md').open().read()
requirements = [requirement.strip() for requirement in Path('requirements.txt').open().readlines()]

setuptools.setup(
    name="length_hpi",
    version="0.0.3",
    author="Christian Bartz, Joseph Bethe",
    author_email="christian.bartz@hpi.de",
    description="A small framework for teaching students about basics of neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    url="https://github.com/HPI-DeepLearning/length",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
