---------------------------------------------------------------------
| FASTCLUSTER: A library for fast, distributed clustering           |
|                                                                   |
| James Philbin <philbinj@gmail.com>                                |
---------------------------------------------------------------------
This is a python library for performing fast, distributed (using MPI)
clustering for very large datasets.

Currently we only support k-means.

---------------------------------------------------------------------
| INSTALLATION                                                      |
---------------------------------------------------------------------
Before installation make sure the following dependencies are met:
- Linux
- CMake >= 2.6.0
- Python >= 2.5 (but < 3.0)
- MPI library (OpenMPI is recommended)
- Numpy (http://numpy.scipy.org/)
- PyTables (http://www.pytables.org/)
- fastann library (http://github.com/philbinj/fastann)

Build the library
> cmake . && make

Install
> make install

---------------------------------------------------------------------
| USAGE                                                             |
---------------------------------------------------------------------
See help(fastcluster.kmeans) for usage.

---------------------------------------------------------------------
| CHANGELOG                                                         |
---------------------------------------------------------------------
v0.1
    - Initial commit
