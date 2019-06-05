from setuptools import setup

setup(
    name='SMI_analysis',
    version='1.0',
    packages=['scipy', 'numpy', 'fabio', 'matplotlib', 'pyFAI', 'git+https://github.com/ronpandolfi/pygix.git'],
    url='',
    license='',
    author='gfreychet',
    author_email='gfreychet@gmail.com',
    description=''
)



install_requires=['scipy', 'Cython', 'pyFAI==0.16.0', 'h5py', 'PySide==1.2.2', 'pyqtgraph', 'QDarkStyle',

                      'Pillow', 'pyfits', 'PyOpenGL', 'PyYAML', 'qtconsole','tifffile','pysftp',

                      'requests','dask','distributed','appdirs','futures','scikit-image','imageio','vispy',

                      'pypaws>=0.8.4','matplotlib', 'astropy'],