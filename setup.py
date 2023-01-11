from setuptools import find_packages, setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='pyfiber',
    packages=find_packages(where='src'),
    version='0.2.11',
    description='Fiber photometry and behavioral data analysis tool',
    author='Matias Ceau',
    author_email="matias@ceau.net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data = {'pyfiber': ['pyfiber.yaml']},
    license='GPLv3',
    url="https://github.com/matias-ceau/pyfiber",
    install_requires=['numpy','scipy','pandas','pyyaml','datetime','seaborn','matplotlib','portion'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers"
    ],
    keywords="fiber photometry, operant behavior",
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest'],
    test_suite='tests',
    package_dir={'': 'src'}
)
