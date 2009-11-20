#ifndef __FASTCLUSTER_KMEANS_HPP
#define __FASTCLUSTER_KMEANS_HPP

namespace fastcluster {

template<class Float>
int
kmeans(void (*load_rows)(void* p, unsigned l, unsigned r, Float* out),
       void* load_rows_p,
       fastann::nn_obj<Float>* (*build_nnobj)(void* p, Float* clusters, unsigned K, unsigned D),
       void* build_nnobj_p,
       Float* clusters,
       unsigned N,
       unsigned D,
       unsigned K,
       unsigned niters,
       int root_rank);

}

#endif
