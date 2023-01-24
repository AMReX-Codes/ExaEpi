#ifndef IO_H_
#define IO_H_

#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"

namespace ExaEpi
{
namespace IO
{

    void writePlotFile (AgentContainer& pc, amrex::iMultiFab& num_residents, amrex::iMultiFab& unit_mf,
                        amrex::iMultiFab& FIPS_mf, amrex::iMultiFab& comm_mf, int step);

}
}

#endif
