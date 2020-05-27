#!/usr/bin/env python

from distutils.core import setup

setup(name='JMCTF',
      version='1.0',
      description='Tools for performing classical hypothesis tests on large joint distributions, powered by tensorflow_probability',
      author='Ben Farmer',
      author_email='ben.farmer@gmail.com',
      url= 'https://github.com/bjfar/jmctf',
      packages=['JMCTF','tests'],
     )
