# =======
# Imports
# =======

from __future__ import print_function
import os
from os.path import join
import sys
import codecs
import subprocess
from .src.nssvie import (
    __package_name__,
    __author__,
    __author_email__,
    __description__, 
    __version__, 
    __url__, 
    __copyright__, 
    __docs_copyright,
    __documentation_url__
)

# ===============
# install package
# ===============

def install_package(package):
    """
    Installs packages using pip.

    Example:

    .. code-block:: python

        >>> install_package('numpy>1.11')

    :param package: Name of package with or without its version pin.
    :type package: string
    """

    subprocess.check_call([sys.executable, "-m", "pip", "install", package])    


# =====================
# Import Setup Packages
# =====================

# Import setuptools
try:
    import setuptools
except ImportError:
    # Install setuptools
    install_package('setuptools')
    import setuptools


# =========
# read file
# =========

def read_file(Filename):
    with codecs.open(Filename, 'r', 'latin') as File:
        return File.read()


# ================
# read file to rst
# ================

def read_file_to_rst(Filename):
    try:
        import pypandoc
        rstname = "{}.{}".format(os.path.splitext(Filename)[0], 'rst')
        pypandoc.convert(
            read_file(Filename), 'rst', format='md', outputfile=rstname)
        with open(rstname, 'r') as f:
            rststr = f.read()
        return rststr
    except ImportError:
        return read_file(Filename)


# ================
# get requirements
# ================

def get_requirements(directory, subdirectory=""):
    """
    Returns a list containing the package requirements given in a file 
    named "requirements.txt" in a subdirectory.
    """

    requirements_filename = join(directory, subdirectory, "requirements.txt")
    requirements_file = open(requirements_filename, 'r')
    requirements = [i.strip() for i in requirements_file.readlines()]

    return requirements


# ====
# main
# ====

def main(argv):

    directory = os.path.dirname(os.path.realpath(__file__))
    package_name = __package_name__

    # Requirements
    requirements = get_requirements(directory)
    test_requirements = get_requirements(directory, subdirectory="tests")
    docs_requirements = get_requirements(directory, subdirectory="docs")

    # ReadMe
    url = __url__
    download_url = url + '/archive/main.zip'
    documentation_url = __documentation_url__
    tracker_url = url + '/issues'

    # Setup
    setuptools.setup(
        name=__package_name__,
        version=__version__,
        author=__author__,
        author_email=__author_email__,
        description=__description__,
        long_description=long_description,
        long_description_content_type='text/x-rst',
        keywords='orthogonal-functions sde volterra-integral-equation ' +
                 'stochastic-differential-equations ' +
                 'stochastic-volterra-integral-equation ' +
                 'numerical-solution block-pulse-functions',
        url=__url__,
        download_url=download_url,
        project_urls={
            "Documentation": __documentation_url__,
            "Source": __url__,
            "Tracker": tracker_url,
        },
        platforms=['Linux', 'OSX', 'Windows'],
        packages=setuptools.find_packages(exclude=[
            'tests.*',
            'tests',
            'examples.*',
            'examples']
        ),
        install_requires=requirements,
        python_requires='>=3.6',
        setup_requires=[
            'setuptools',
            'pytest-runner'],
        tests_require=[
            'pytest',
            'pytest-cov'],
        include_package_data=True,
        zip_safe=False,
        extras_require={
            'test': test_requirements,
            'docs': docs_requirements,
        },
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    )


# ===========
# script main
# ===========

if __name__ == "__main__":
    main(sys.argv)
