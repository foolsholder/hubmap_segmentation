from setuptools import find_packages, setup

setup(
    name="hubmap_segmentation",
    packages=find_packages(include=["hubmap_segmentation"]),
    version="0.1.0",
    author="Me",
    license="MIT",
    install_requires=[],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)
