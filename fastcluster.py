#!/usr/bin/env python

import os, sys
import ctypes
import tables
import numpy as np
import numpy.random as npr

libfastann = ctypes.CDLL('libfastann.so')
try:
    libfastcluster = ctypes.CDLL('libfastcluster.so')
except OSError, e:
    libfastcluster = ctypes.CDLL('./libfastcluster.so')


def get_suffix(dtype):
    return {np.dtype('u1') : 'c',
            np.dtype('f4') : 's',
            np.dtype('f8') : 'd'}[dtype]

class nn_obj_exact_builder(object):
    def __init__(self, dtype):
        self.dtype = dtype
        self.suffix = get_suffix(dtype)

    def build_nn_obj(self, p, clusters, K, D):
        ptr = getattr(libfastann, "fastann_nn_obj_build_exact_" + self.suffix)(clusters, K, D)

        return ptr

class nn_obj_approx_builder(object):
    def __init__(self, dtype, ntrees, nchecks):
        self.dtype = dtype
        self.suffix = get_suffix(dtype)
        self.ntrees = ntrees
        self.nchecks = nchecks

    def build_nn_obj(self, p, clusters, K, D):
        ptr = getattr(libfastann, "fastann_nn_obj_build_kdtree_" + self.suffix)\
                (ctypes.c_void_p(clusters), ctypes.c_uint(K), ctypes.c_uint(D), 
                 ctypes.c_uint(self.ntrees), ctypes.c_uint(self.nchecks))

        return ptr

class hdf5_wrap(object):
    def __init__(self, pnts_obj, dt):
        self.dt = dt
        self.pnts_obj = pnts_obj

    def read_rows(self, p, l, r, out):
        pnts = self.pnts_obj[l:r].astype(self.dt)

        pnts_ptr = pnts.ctypes.data_as(ctypes.c_void_p)

        ctypes.memmove(ctypes.c_void_p(out), pnts_ptr, pnts.dtype.itemsize*(r-l)*pnts.shape[1])

def kmeans(clst_fn,
           pnts_fn, 
           K, 
           niters = 30, 
           approx = True,
           ntrees = 8, 
           nchecks = 784, 
           checkpoint = True,
           seed = 42):
    """
    Runs the distributed approximate k-means algorithm.

    Params
    ------
    clst_fn : string
        HDF5 filename for the cluster output
    pnts_fn : string
        HDF5 filename for the points to cluster
    K : int
        Number of clusters
    niters : int (30)
        Number of iterations
    approx : bool (True)
        Exact or approximate nn
    ntrees : int (8)
        Size of the k-d forest
    nchecks : int (768)
        Number of point distances to compute per query
    checkpoint : bool (True)
        Whether to checkpoint
    seed : int (42)
        Random seed
    """
    errc = libfastcluster.safe_init()
    if errc: raise RuntimeError, 'problem with mpi_init'

    npr.seed(seed)
    # Probe for datatype and dimensionality
    pnts_fobj = tables.openFile(pnts_fn, 'r')
    for pnts_obj in pnts_fobj.walkNodes('/', classname = 'Array'):
        break

    N = pnts_obj.shape[0]
    D = pnts_obj.shape[1]
    dtype = pnts_obj.atom.dtype
    
    if dtype not in [np.dtype('u1'), np.dtype('f4'), np.dtype('f8')]:
        raise TypeError, 'Datatype %s not supported' % dtype

    if dtype == np.dtype('u1'):
        dtype = np.dtype('f4')

    if approx:
        nn_builder = nn_obj_approx_builder(dtype, ntrees, nchecks)
    else:
        nn_builder = nn_obj_exact_builder(dtype)
    
    pnt_loader = hdf5_wrap(pnts_obj, dtype)

    # Callbacks
    LOAD_ROWS_FUNC = \
        ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, 
                         ctypes.c_void_p)

    NN_BUILDER_FUNC = \
        ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                         ctypes.c_uint, ctypes.c_uint)

    load_rows_func = LOAD_ROWS_FUNC(pnt_loader.read_rows)
    nn_builder_func = NN_BUILDER_FUNC(nn_builder.build_nn_obj)

    # Space for the clusters
    clusters = np.empty((K, D), dtype = dtype)
    clusters_ptr = clusters.ctypes.data_as(ctypes.c_void_p)
    # Initialize the clusters
    if libfastcluster.rank() == 0:
        sys.stdout.write('Sampling cluster centers...')
        sys.stdout.flush()
        pnts_inds = np.arange(N)
        npr.shuffle(pnts_inds)
        pnts_inds = pnts_inds[:K]
        pnts_inds = np.sort(pnts_inds)
        for i,ind in enumerate(pnts_inds):
            clusters[i] = pnts_obj[ind]
            if not (i%(K/100)):
                sys.stdout.write('\r[%07d/%07d]' % (i, K))
                sys.stdout.flush()
        sys.stdout.write('Done...')
        sys.stdout.flush()

    if checkpoint:
        chkpnt_fn = clst_fn + '.chkpnt'
    else:
        chkpnt_fn = ''

    getattr(libfastcluster, "kmeans_" + get_suffix(dtype))\
       (load_rows_func,
        ctypes.c_void_p(0),
        nn_builder_func,
        ctypes.c_void_p(0),
        clusters_ptr,
        ctypes.c_uint(N),
        ctypes.c_uint(D),
        ctypes.c_uint(K),
        ctypes.c_uint(niters),
        ctypes.c_int(0),
        ctypes.c_char_p(chkpnt_fn))

    # All done, save the clusters
    if libfastcluster.rank() == 0:
        filters = tables.Filters(complevel = 1, complib = 'zlib')
        clst_fobj = tables.openFile(clst_fn, 'w')
        clst_obj = \
            clst_fobj.createCArray(clst_fobj.root, 'clusters',
                                   tables.Atom.from_dtype(dtype), clusters.shape,
                                   filters = filters)
        clst_obj[:] = clusters

        clst_fobj.close()

        if chkpnt_fn != '':
            try:
                os.remove(chkpnt_fn)
            except OSError, e:
                pass

if __name__ == "__main__":
    N = 1000000
    D = 128
    K = 10000
    npr.seed(42)

    libfastcluster.safe_init()

    if libfastcluster.rank() == 0:
        pnts = npr.randn(N, D).astype('f4')
        pnts_fobj = tables.openFile('pnts.h5','w')
        pnts_fobj.createArray(pnts_fobj.root, 'pnts', pnts)
        pnts_fobj.close()
        del pnts
    libfastcluster.barrier()

    kmeans('clst.h5',
           'pnts.h5',
           K,
           30)


