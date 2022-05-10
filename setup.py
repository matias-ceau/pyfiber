from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyfiber',
    packages=find_packages(where='pyfiber'),
    version='0.1.0',
    description='Fiber photometry and behavioral data coupling',
    author='Matias Ceau',
    author_email="matias@ceau.net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url="https://github.com/matias-ceau/pyfiber",
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    package_dir={"": "pyfiber"}
)
