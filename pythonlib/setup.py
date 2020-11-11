from setuptools import find_packages, setup
from distutils.core import setup, Extension
import os

path = os.getcwd()
setup(
    name='ardis',
    packages=['ardis'],
    url="none.none",
    version='1.0.0',
    description='Library for simulating reaction-diffusions systems',
    author='Hedi Sellami',
    author_email='hedi@sellami.dev',
    install_requires=['numpy','scipy','matplotlib'],
    # ext_modules = [Extension('ardisLib',sources=[],libraries=[os.getcwd()+'/ardisLib.so'])],
    license='MIT',
    python_requires='>=3.0',
    package_data={'ardis': ['ardisLib.so']},
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest'],
    # test_suite='ardis',
)