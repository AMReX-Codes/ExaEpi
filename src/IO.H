/*! @file IO.H
    \brief Defines the #ExaEpi::IO namespace
*/

#ifndef IO_H_
#define IO_H_

#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "DemographicData.H"

#include <string>

namespace ExaEpi
{

/*! \brief Namespace for IO functions; note that this namespace contains that write
           plotfiles and diagnostics data */
namespace IO
{

    void writePlotFile (const AgentContainer& pc, const amrex::iMultiFab& num_residents, const amrex::iMultiFab& unit_mf,
                        const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& comm_mf, const amrex::Real cur_time, const int step);

    void writeFIPSData (const AgentContainer& pc, const amrex::iMultiFab& unit_mf,
                        const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& comm_mf,
                        const DemographicData& demo, const std::string& prefix, const int step);
}
}

#endif
