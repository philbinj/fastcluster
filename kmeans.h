#ifndef __FASTCLUSTER_KMEANS_H
#define __FASTCLUSTER_KMEANS_H

int
kmeans_s(
         void (*load_rows)(void* p, unsigned l, unsigned r, float* out),
         void* load_rows_p,
         void* (*build_nnobj)(void*, float*, unsigned, unsigned),
         void* build_nnobj_p,
         float* clusters,
         unsigned N,
         unsigned D,
         unsigned K,
         unsigned niters,
         int root_rank,
         char* chkpnt_fn);

int
kmeans_d(
         void (*load_rows)(void* p, unsigned l, unsigned r, double* out),
         void* load_rows_p,
         void* (*build_nnobj)(void*, double*, unsigned, unsigned),
         void* build_nnobj_p,
         double* clusters,
         unsigned N,
         unsigned D,
         unsigned K,
         unsigned niters,
         int root_rank,
         char* chkpnt_fn);

#endif
