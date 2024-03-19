from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='src',
    version=__version__,
    author='xx',
    author_email='example@gmail.com',
    url='-',
    license='MIT',
    packages=find_packages(include=['src']),
    zip_safe=False,
)
