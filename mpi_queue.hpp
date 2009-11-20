#ifndef __FASTCLUSTER_MPI_QUEUE_HPP
#define __FASTCLUSTER_MPI_QUEUE_HPP

#include <iosfwd>
#include <vector>

#include <mpi.h>

namespace fastcluster {

class
work_item
{
public:
    virtual ~work_item() { }

    virtual void serialize(std::ostream& ostr) const = 0;

    virtual work_item* unserialize(std::istream& istr) const = 0;
};

class
result_item
{
public:
    virtual ~result_item() { }

    virtual void serialize(std::ostream& ostr) const = 0;

    virtual result_item* unserialize(std::istream& istr) const = 0;
};

class
work_functor
{
public:
    virtual ~work_functor() { }

    virtual result_item* do_work(const work_item* work) = 0;
};

/**
 * A static MPI queue which evenly splits the work among the
 * participating nodes.
 *
 * The results are only accumulated on the \c root_rank node.
 */
void
mpi_static_queue(MPI_Comm comm,
                 int root_rank,
                 work_functor* workf,
                 const std::vector<work_item*>& work,
                 std::vector<result_item*>* results = 0,
                 const result_item* dummy_result = 0);

void
mpi_dynamic_queue(MPI_Comm comm,
                  int root_rank,
                  work_functor* workf,
                  const std::vector<work_item*>& work,
                  std::vector<result_item*>* results = 0,
                  const result_item* dummy_result = 0);

}

#endif
