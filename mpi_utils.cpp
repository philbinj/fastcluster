/**
 * Some safe mpi routines so we don't require any particular external
 * python library.
 */

#include <mpi.h>

extern "C"
int
safe_init()
{
    int flag;
    MPI_Initialized(&flag);

    if (!flag) {
        // We should init.
        int argc = 0;
        char** argv = 0;

        int errc = MPI_Init(&argc, &argv);

        if (errc != MPI_SUCCESS) return 1;
    }

    return 0;
}

extern "C"
int
rank()
{
    int ret;
    MPI_Comm_rank(MPI_COMM_WORLD, &ret);
    return ret;
}

extern "C"
int
size()
{
    int ret;
    MPI_Comm_size(MPI_COMM_WORLD, &ret);
    return ret;
}

extern "C"
void
barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
}
