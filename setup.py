from setuptools import setup, find_packages

setup(
    name='SMI_analysis',
    version='1.0',
    py_modules=['SMI_beamline', 'remesh', 'stitch', 'integrate1D', 'Detector'],
    install_requires=['numpy', 'scipy', 'pyFAI', 'pygix==0.1.4a0',],
    dependency_links=['git+ssh://git@github.com/ronpandolfi/pygix.git#egg=pygix-0.1.4a0',],
    url='',
    license='',
    author='gfreychet',
    author_email='gfreychet@gmail.com',
    description='SMI analysis tools'
)