#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "mpi_queue.hpp"
#include "whetstone.hpp"

namespace fastcluster {

void
mpi_static_queue(MPI_Comm comm,
                 int root_rank,
                 work_functor* workf,
                 const std::vector<work_item*>& work,
                 std::vector<result_item*>* results,
                 const result_item* dummy_result)
{
    int rank, nprocs;
    size_t proc_l, proc_r, remote_l, remote_r;
    int transmit_sz;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    // Use whetstone to estimate the speed (in FLOPS) of each rank.
    double cpu_time;
    double cpu_flops;
    whetstone(cpu_flops, cpu_time, 0.0);

    std::vector<unsigned> rank_flops(nprocs, 0.0);
    rank_flops[rank] = (unsigned)(cpu_flops/1.e06);

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                  &rank_flops[0], 1, MPI_UNSIGNED, MPI_COMM_WORLD);

    std::vector<unsigned> accum_flops(nprocs + 1, 0.0);
    for (int i=0; i < nprocs; ++i) accum_flops[i+1] = accum_flops[i] + rank_flops[i];

    proc_l = (accum_flops[rank] * work.size())/accum_flops[nprocs];
    proc_r = (accum_flops[rank + 1] * work.size())/accum_flops[nprocs];
    
    std::vector<result_item*> local_results;
    for (size_t i=proc_l; i < proc_r; ++i) {
        result_item* res_i = workf->do_work(work[i]);

        local_results.push_back(res_i);
    }

    // If we don't have to return results, then we're all done!
    if (!results) return;

    if (rank == root_rank) {
        (*results).resize(work.size(), 0);
        // Copy my local_results to the right part of the results array.
        std::copy(local_results.begin(), local_results.end(), results->begin() + proc_l);
        for (int remote_rank = 0; remote_rank < nprocs; ++remote_rank) {
            if (remote_rank == rank) continue;

            remote_l = (accum_flops[remote_rank] * work.size())/accum_flops[nprocs];
            remote_r = (accum_flops[remote_rank] * work.size())/accum_flops[nprocs];

            for (size_t remote_i = remote_l; remote_i < remote_r; ++remote_i) {
                MPI_Recv(&transmit_sz, 1, MPI_INT, remote_rank, 0, comm, 0);

                if (transmit_sz) {
                    char* receive_buf = new char[transmit_sz];
                    MPI_Recv(receive_buf, transmit_sz, MPI_CHAR, remote_rank, 0, comm, 0);

                    std::istringstream iss(std::string(receive_buf, transmit_sz));

                    (*results)[remote_i] = dummy_result->unserialize(iss);

                    delete[] receive_buf;
                }
            }
        }
    }
    else {
        for (size_t i = 0; i < (proc_r - proc_l); ++i) {
            if (local_results[i]) {
                std::ostringstream oss;
                local_results[i]->serialize(oss);
                std::string oss_str = oss.str();

                int transmit_sz = (int)oss_str.size();
                MPI_Send(&transmit_sz, 1, MPI_INT, root_rank, 0, comm);
                MPI_Send(&oss_str[0], (int)oss_str.size(), MPI_CHAR, root_rank, 0, comm);

                delete local_results[i];
            }
            else {
                int transmit_sz = 0;
                MPI_Send(&transmit_sz, 1, MPI_INT, root_rank, 0, comm);
            }
        }
    }
    MPI_Barrier(comm);
}

void
mpi_dynamic_queue(MPI_Comm comm,
                  int root_rank,
                  work_functor* workf,
                  const std::vector<work_item*>& work,
                  std::vector<result_item*>* results,
                  const result_item* dummy_result)
{
    int rank, nprocs;

    MPI_Status status;
    int transmit_sz;
    char* transmit_buf;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (nprocs < 2) {
        return mpi_static_queue(comm, root_rank, workf, work, results, dummy_result);
    }

    int nmap = (int)work.size();
    int doneflag = -1;
    int ndone = 0;
    int itask = 0;

    if (rank == root_rank) {
        // Get everybody working...
        for (int proc = 0; proc < nprocs; ++proc) {
            if (proc == root_rank) continue;

            if (itask < nmap) {
                MPI_Send(&itask, 1, MPI_INT, proc, 0, comm);
                itask++;
            }
            else {
                MPI_Send(&doneflag, 1, MPI_INT, proc, 0, comm);
                ndone++;
            }
        }
        if (results)
            results->resize(work.size(), 0);
        while (ndone < nprocs-1) {
            // Receive a result from the processor.
            int tasknum, proc;
            MPI_Recv(&tasknum, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &status);
            proc = status.MPI_SOURCE;
            
            if (results && dummy_result) {
                MPI_Recv(&transmit_sz, 1, MPI_INT, proc, 0, comm, 0);
                transmit_buf = new char[transmit_sz];
                MPI_Recv(transmit_buf, transmit_sz, MPI_CHAR, proc, 0, comm, 0);

                std::istringstream iss(std::string(transmit_buf, transmit_sz));
                
                (*results)[tasknum] = dummy_result->unserialize(iss);

                delete[] transmit_buf;
            }

            // Send some new work.
            if (itask < nmap) {
                MPI_Send(&itask, 1, MPI_INT, proc, 0, comm);
                itask++;
            }
            else {
                MPI_Send(&doneflag, 1, MPI_INT, proc, 0, comm);
                ndone++;
            }
        }
    }
    else {
        while (1) {
            int itask;
            MPI_Recv(&itask, 1, MPI_INT, root_rank, 0, comm, 0);
            if (itask < 0) break;

            result_item* res = workf->do_work(work[itask]);

            MPI_Send(&itask, 1, MPI_INT, root_rank, 0, comm);
            
            if (results && dummy_result) {
                std::ostringstream oss;
            
                res->serialize(oss);

                std::string oss_str = oss.str();
                const char* oss_cstr = oss_str.c_str();
                int oss_sz = (int)oss_str.size();

                MPI_Send(&oss_sz, 1, MPI_INT, root_rank, 0, comm);
                MPI_Send((void*)oss_cstr, oss_sz, MPI_CHAR, root_rank, 0, comm);
            }

            delete res;
        }
    }
    MPI_Barrier(comm);
}

}
