from setuptools import setup, find_packages
from glob import glob

setup(
    name='speech_command_classifier',
    packages=find_packages(include=['speech_command_classifier']),
    url='',
    python_requires='~=3.8.8',
    install_requires=[
        'numpy',
    ],
    dependency_links=[
    ],
    setup_requires=[
        'setuptools_scm',
    ],
    license='License :: Other/Proprietary License',
    author='',
    author_email='',
    description='',
    classifiers=[
    ],
    scripts=glob('bin/*'),
)
