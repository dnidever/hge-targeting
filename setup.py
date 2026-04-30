#!/usr/bin/env python

from distutils.core import setup

setup(name='hgetargeting',
      version='1.0.0',
      description='AS5 HGE targeting',
      author='David Nidever and others',
      author_email='dnidever@montana.edu',
      url='https://github.com/sdss/hge-targeting/',
      packages=['hgetargeting'],
      package_dir={'':'python'},
      requires=['numpy','astropy','scipy'],
      include_package_data=True
)
