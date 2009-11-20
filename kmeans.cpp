#include <cmath>
#include <cstdlib>
#include <vector>
#include <fastann/fastann.hpp>
#include "randomkit.h"
#include "mpi_queue.hpp"
#include "kmeans.hpp"

namespace fastcluster {

double node_build_time = 0.0;
double node_load_time = 0.0;
double node_search_time = 0.0;
double node_reduce_time = 0.0;
double node_bcast_time = 0.0;
double node_queue_time = 0.0;
double node_total_time = 0.0;

class
kmeans_work_item : public work_item
{
public:
    kmeans_work_item()
     : l_(0), r_(0)
    { }
    kmeans_work_item(unsigned l, unsigned r)
     : l_(l), r_(r)
    { }

    virtual void serialize(std::ostream& ostr) const
    {
        ostr.write((const char*)&l_, sizeof(unsigned));
        ostr.write((const char*)&r_, sizeof(unsigned));
    }

    virtual kmeans_work_item* unserialize(std::istream& istr) const
    {
        kmeans_work_item* ret = new kmeans_work_item;

        istr.read((char*)ret->l_, sizeof(unsigned));
        istr.read((char*)ret->r_, sizeof(unsigned));

        return ret;
    }

    unsigned get_l() const { return l_; }
    unsigned get_r() const { return r_; }

private:
    unsigned l_;
    unsigned r_;
};

template<class Float>
class
kmeans_work_functor : public work_functor
{
public:
    kmeans_work_functor(fastann::nn_obj<Float>* nno,
                        Float* clst_sums,
                        unsigned* clst_sums_n,
                        Float* clst_dist,
                        unsigned D,
                        void (*load_rows)(void*,unsigned,unsigned,Float*),
                        void* load_rows_p)
     : nno_(nno), clst_sums_(clst_sums), clst_sums_n_(clst_sums_n),
       clst_dist_(clst_dist), D_(D), load_rows_(load_rows), 
       load_rows_p_(load_rows_p)
    { }

    virtual
    result_item*
    do_work(const work_item* work) 
    {
        double t1, t2;
        const kmeans_work_item* wi = static_cast<const kmeans_work_item*>(work);

        unsigned l = wi->get_l();
        unsigned r = wi->get_r();

        points_.resize((r-l)*D_);
        argmins_.resize((r-l));
        mins_.resize((r-l));

        t1 = MPI_Wtime();
        load_rows_(load_rows_p_, l, r, &points_[0]);
        t2 = MPI_Wtime(); node_load_time += (t2 - t1);

        t1 = MPI_Wtime();
        nno_->search_nn(&points_[0], r-l, &argmins_[0], &mins_[0]);
        t2 = MPI_Wtime(); node_search_time += (t2 - t1);

        for (unsigned i=0; i < (r-l); ++i) {
            unsigned k = argmins_[i];
            for (unsigned d=0; d < D_; ++d) {
                clst_sums_[k*D_ + d] += points_[i*D_ + d];
            }
            clst_sums_n_[k] += 1;
            (*clst_dist_) += mins_[i];
        }

        return 0;
    }
private:
    fastann::nn_obj<Float>* nno_;
    Float* clst_sums_;
    unsigned* clst_sums_n_;
    Float* clst_dist_;
    unsigned D_;
    void (*load_rows_)(void* p, unsigned l, unsigned r, Float* out);
    void* load_rows_p_;
    std::vector<Float> points_;
    std::vector<unsigned> argmins_;
    std::vector<Float> mins_;
};

template<class Data>
struct data_mpi_map
{
};

template<>
struct data_mpi_map<float>
{
    static MPI_Datatype type() { return MPI_FLOAT; }
};

template<>
struct data_mpi_map<double>
{
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

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
       int root_rank,
       char* chkpnt_fn)
{
    unsigned probe_size = 500;
    static const int dynamic_nprocs_thresh = 30;
    unsigned block_size = 50000; // 50K rows blocksize

    int rank, nprocs;

    double t1, t2;
    double tt1, tt2;

    // A few sanity checks
    if (D > N) {
        fprintf(stderr, "WARNING: kmeans: D > N - you might need to transpose\n");
    }

    if (K > N) {
        fprintf(stderr, "FATAL: kmeans: K > N - can't have more clusters than data\n");
    }

    probe_size = std::min(probe_size, N);

    // Quick check to see if we're initialized
    int init_flag;
    MPI_Initialized(&init_flag);
    if (!init_flag) {
        fprintf(stderr, "FATAL: kmeans: MPI not initialized!!!\n");
        return 1;
    }

    rk_state state;
    rk_seed(42, &state);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<Float> nclusters(K*D);
    std::vector<unsigned> ncluster_cnts(K);
    Float cluster_dists;

    unsigned iter = 0;

    // --- Checkpoint load ---
    if (rank == root_rank) {
        if (chkpnt_fn) {
            FILE* chkpnt_fobj = fopen(chkpnt_fn, "rb");
            if (chkpnt_fobj) {
                int success = 0;
                do {
                    if (fread(&nclusters[0], sizeof(Float), K*D, chkpnt_fobj)!=(K*D))
                        break;

                    if (fread(&iter, sizeof(unsigned), 1, chkpnt_fobj)!=1)
                        break;

                    success = 1;
                }
                while (0);

                fclose(chkpnt_fobj);
                if (!success) {
                    iter = 0;
                }
                else {
                    std::copy(nclusters.begin(), nclusters.end(), clusters);
                    printf("Restarting from checkpoint, iter = %d\n", iter);
                }
            }
        }
        printf("N = %d, K = %d, D = %d\n", N, K, D);
        printf("%6s %15s %6s %6s %6s %6s %6s %6s %10s | %13s\n", "ITER", "SSD", "BUILD", "LOAD", "SEARCH", "REDUCE", "BCAST", "QUEUE", "TIME", "ACC");
        printf("-------------------------------------------------------------------------------------------\n");
        fflush(stdout);
    }
    // --- ---
    MPI_Bcast(&iter, 1, MPI_UNSIGNED, root_rank, MPI_COMM_WORLD);

    for ( ; iter < niters; ++iter) {
        // Distribute clusters to all nodes.
        tt1 = MPI_Wtime();
        t1 = MPI_Wtime();
        MPI_Bcast(clusters, D*K, data_mpi_map<Float>::type(), root_rank, MPI_COMM_WORLD);
        t2 = MPI_Wtime(); node_bcast_time += (t2 - t1);

        // New cluster sums.
        std::fill(nclusters.begin(), nclusters.end(), Float(0));
        std::fill(ncluster_cnts.begin(), ncluster_cnts.end(), 0);
        cluster_dists = Float(0);

        // Queue of points
        std::vector<work_item*> work_queue;
        for (unsigned bl = 0; bl < N; bl += block_size) {
            unsigned br = std::min(bl + block_size, N);
            work_queue.push_back(new kmeans_work_item(bl, br));
        }

        // Build the k-d trees
        t1 = MPI_Wtime();
        fastann::nn_obj<Float>* nno = build_nnobj(build_nnobj_p, clusters, K, D);
        t2 = MPI_Wtime(); node_build_time += (t2 - t1);

        kmeans_work_functor<Float>* wfunc = 
            new kmeans_work_functor<Float>(
                                    nno,
                                    &nclusters[0],
                                    &ncluster_cnts[0],
                                    &cluster_dists,
                                    D,
                                    load_rows,
                                    load_rows_p);

        t1 = MPI_Wtime();
        if (nprocs > dynamic_nprocs_thresh) {
            mpi_dynamic_queue(MPI_COMM_WORLD, root_rank, wfunc, work_queue);
        }
        else {
            mpi_static_queue(MPI_COMM_WORLD, root_rank, wfunc, work_queue);
        }
        t2 = MPI_Wtime(); node_queue_time += (t2 - t1);
        
        // Sum up all the results.
        t1 = MPI_Wtime();
        if (rank == root_rank) {
            MPI_Reduce(MPI_IN_PLACE, &nclusters[0], K*D, data_mpi_map<Float>::type(), MPI_SUM, root_rank, MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &ncluster_cnts[0], K, MPI_UNSIGNED, MPI_SUM, root_rank, MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &cluster_dists, 1, data_mpi_map<Float>::type(), MPI_SUM, root_rank, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(&nclusters[0], 0, K*D, data_mpi_map<Float>::type(), MPI_SUM, root_rank, MPI_COMM_WORLD);
            MPI_Reduce(&ncluster_cnts[0], 0, K, MPI_UNSIGNED, MPI_SUM, root_rank, MPI_COMM_WORLD);
            MPI_Reduce(&cluster_dists, 0, 1, data_mpi_map<Float>::type(), MPI_SUM, root_rank, MPI_COMM_WORLD);
        }
        t2 = MPI_Wtime(); node_reduce_time += (t2 - t1);
        tt2 = MPI_Wtime(); node_total_time += (tt2 - tt1);

        // Stats
        double tot_build_time = 0.0, tot_load_time = 0.0, tot_search_time = 0.0, 
               tot_reduce_time = 0.0, tot_bcast_time = 0.0, tot_queue_time = 0.0,
               tot_total_time = 0.0;
        MPI_Reduce(&node_build_time, &tot_build_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
        MPI_Reduce(&node_load_time, &tot_load_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
        MPI_Reduce(&node_search_time, &tot_search_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
        MPI_Reduce(&node_reduce_time, &tot_reduce_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
        MPI_Reduce(&node_bcast_time, &tot_bcast_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
        MPI_Reduce(&node_queue_time, &tot_queue_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
        MPI_Reduce(&node_total_time, &tot_total_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);

        // Average the cluster sums.
        if (rank == root_rank) {
            for (unsigned k=0; k < K; ++k) {
                if (!ncluster_cnts[k]) {
                    // If there's an empty cluster we replace it with a random point.
                    printf("Warning: Cluster %d is empty!\n", k);
                    ncluster_cnts[k] = 1;
                    unsigned rand_point_ind = rk_interval(N-1, &state);
                    load_rows(load_rows_p, rand_point_ind, rand_point_ind+1, &nclusters[k*D]);
                }
            }
            for (unsigned k=0; k < K; ++k) {
                for (unsigned d=0; d < D; ++d) {
                    nclusters[k*D + d] /= ncluster_cnts[k];
                }
            }
            
            // Estimate accuracy.
            double acc = 0.0;
            double acc_err = 0.0;
            {
                std::vector<Float> temp_pnt(D);
                fastann::nn_obj<Float>* nno_exact = fastann::nn_obj_build_exact(clusters, K, D);

                Float min_approx, min_true;
                unsigned argmin_approx, argmin_true;

                double p = 0.0;
                for (unsigned i=0; i < probe_size; ++i) {
                    unsigned rand_point_ind = rk_interval(N-1, &state);
                    load_rows(load_rows_p, rand_point_ind, rand_point_ind+1, &temp_pnt[0]);

                    nno->search_nn(&temp_pnt[0], 1, &argmin_approx, &min_approx);
                    nno_exact->search_nn(&temp_pnt[0], 1, &argmin_true, &min_true);       

                    if (argmin_true == argmin_approx) p += 1.0;
                }
                p /= probe_size;

                acc = p;
                acc_err = (1.0/sqrt(probe_size))*sqrt((p*(1.0-p)));
            }
            
            std::copy(nclusters.begin(), nclusters.end(), clusters);

            tt2 = MPI_Wtime();
            printf("%6d %15e %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%% %9.1fs | %4.2f%%+-%4.2f%%\n",
                   iter,
                   cluster_dists,
                   100.0*tot_build_time/tot_total_time,
                   100.0*tot_load_time/tot_total_time,
                   100.0*tot_search_time/tot_total_time,
                   100.0*tot_reduce_time/tot_total_time,
                   100.0*tot_bcast_time/tot_total_time,
                   100.0*(tot_queue_time - tot_search_time)/tot_total_time,
                   tt2 - tt1,
                   100.0*acc,
                   100.0*acc_err);

            // --- Checkpoint save ---
            if ((iter != niters-1) && chkpnt_fn) { // Checkpoint save.
                FILE* chkpnt_fobj = fopen(chkpnt_fn, "wb");
                if (chkpnt_fobj) {
                    int success = 0;
                    unsigned liter = iter+1;
                    do {
                        if (fwrite(clusters, sizeof(Float), K*D, chkpnt_fobj)!=(K*D))
                            break;

                        if (fwrite(&liter, sizeof(unsigned), 1, chkpnt_fobj)!=1)
                            break;

                        success = 1;
                    }
                    while (0);

                    fclose(chkpnt_fobj);
                    if (!success) {
                        // We should try to delete it
                        remove(chkpnt_fn);
                    }
                }
                else {
                    fprintf(stderr, "WARNING: kmeans: Can't save checkpoint...\n");
                }
            }
            // --- ---
        }

        delete wfunc;
        delete nno;

        for (size_t i=0; i < work_queue.size(); ++i) delete work_queue[i];
    }

    return 0;
}

}

typedef fastann::nn_obj<float>*(*build_nnobj_s_t)(void*, float*, unsigned, unsigned);
typedef fastann::nn_obj<double>*(*build_nnobj_d_t)(void*, double*, unsigned, unsigned);

extern "C"
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
         char* chkpnt_fn)
{
    return
        fastcluster::kmeans<float>(load_rows, load_rows_p, 
                                   (build_nnobj_s_t)build_nnobj, build_nnobj_p,
                                   clusters, N, D, K, niters,
                                   root_rank, chkpnt_fn);
}

extern "C"
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
         char* chkpnt_fn)
{
    return 
        fastcluster::kmeans<double>(load_rows, load_rows_p, 
                                    (build_nnobj_d_t)build_nnobj, build_nnobj_p,
                                    clusters, N, D, K, niters,
                                    root_rank, chkpnt_fn);
}
