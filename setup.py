# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# list requirements for setuptools
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = []
    for name in f.readlines():
        if not name.startswith('--'):
            requirements.append(name.rstrip())

setup(
    name='pm25predict',
    version='0.4',
    description='PM2.5predict',
    author='twairball',
    author_email='twairball@yahoo.com',
    packages=find_packages(exclude=['contrib', 'docs', 'test*']),
    install_requires=requirements,
)