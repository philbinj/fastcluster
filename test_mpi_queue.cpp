#include <cassert>
#include <stdio.h>

#include "mpi_queue.hpp"

class test_work_item : public fastcluster::work_item
{
public:
    test_work_item()
      : number(0)
    { }
    test_work_item(int num)
      : number(num)
    { }

    virtual
    void
    serialize(std::ostream& ostr) const
    {
        ostr.write((const char*)&number, sizeof(int));
    }

    virtual
    test_work_item*
    unserialize(std::istream& istr) const
    {
        test_work_item* ret = new test_work_item;

        istr.read((char*)&ret->number, sizeof(int));

        return ret;
    }

    virtual
    int
    get_number() const { return number; }

private:
    int number;
};

class test_result_item : public fastcluster::result_item
{
public:
    test_result_item()
     : number(0)
    { }

    test_result_item(int num)
     : number(num)
    { }

    virtual
    void
    serialize(std::ostream& ostr) const
    {
        ostr.write((const char*)&number, sizeof(int));
    }

    virtual
    test_result_item*
    unserialize(std::istream& istr) const
    {
        test_result_item* ret = new test_result_item;
        istr.read((char*)&ret->number, sizeof(int));

        return ret;
    }

    virtual
    int
    get_number() const { return number; }

private:
    int number;
};

class test_work_functor : public fastcluster::work_functor
{
public:
    virtual
    fastcluster::result_item*
    do_work(const fastcluster::work_item* work) 
    {
        const test_work_item* witem = static_cast<const test_work_item*>(work);

        //printf("%d\n", witem->get_number());

        return new test_result_item(witem->get_number());
    }
};

class test_work_functor2 : public fastcluster::work_functor
{
public:
    test_work_functor2()
     : sum(0)
    { }

    virtual
    fastcluster::result_item*
    do_work(const fastcluster::work_item* work)
    {
        const test_work_item* witem = static_cast<const test_work_item*>(work);

        sum += witem->get_number();

        return 0;
    }
    
    int
    get_sum() const
    { return sum; }

private:
    int sum;
};

template<class QueueFunc>
int
test_with_items(QueueFunc queuefunc)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    std::vector<fastcluster::work_item*> work_queue;
    std::vector<fastcluster::result_item*> result_queue;
    for (int i=0; i < 10000; ++i) {
        work_queue.push_back(new test_work_item(i));
    }
    test_work_functor* func = new test_work_functor;
    test_result_item* dummy_result = new test_result_item;

    queuefunc(MPI_COMM_WORLD, 0, func, work_queue, &result_queue, dummy_result);

    delete func;
    delete dummy_result;

    for (int i=0; i < (int)work_queue.size(); ++i) {
        delete work_queue[i];
    }
    if (rank == 0) {
        for (int i=0; i < (int)result_queue.size(); ++i) {
            if (result_queue[i]) {
                if (static_cast<test_result_item*>(result_queue[i])->get_number() != i) {
                    return -1;
                }
                //printf("%d\n", static_cast<test_result_item*>(result_queue[i])->get_number());
                delete result_queue[i];
            }
        }
    }
    return 0;
}

template<class QueueFunc>
int
test_without_items(QueueFunc queuefunc)
{
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    std::vector<fastcluster::work_item*> work_queue;
    
    for (int i=0; i < 10000; ++i) {
        work_queue.push_back(new test_work_item(i));
    }
    test_work_functor2* func = new test_work_functor2;

    queuefunc(MPI_COMM_WORLD, 0, func, work_queue, 0, 0);

    int local_sum = 0;
    local_sum += func->get_sum();

    MPI_Allreduce(MPI_IN_PLACE, &local_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (local_sum == 49995000) return 0;
    else return -1;
}

int
main(int argc, char** argv)
{
    int errc;
    MPI_Init(&argc, &argv);

    errc = test_with_items(fastcluster::mpi_static_queue);
    if (errc) { printf("WI_STATIC: FAILED\n"); return -1; }
    else { printf("WI_STATIC: PASSED\n"); }
    
    errc = test_with_items(fastcluster::mpi_dynamic_queue);
    if (errc) { printf("WI_DYNAMIC: FAILED\n"); return -1; }
    else { printf("WI_DYNAMIC: PASSED\n"); }
    
    errc = test_without_items(fastcluster::mpi_static_queue);
    if (errc) { printf("WO_STATIC: FAILED\n"); return -1; }
    else { printf("WO_STATIC: PASSED\n"); }

    errc = test_without_items(fastcluster::mpi_dynamic_queue);
    if (errc) { printf("WO_DYNAMIC: FAILED\n"); return -1; }
    else { printf("WO_DYNAMIC: PASSED\n"); }
    
    MPI_Finalize();

    return 0;
}
