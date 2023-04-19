#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include "DemographicData.H"
#include "Utils.H"
#include "AgentContainer.H"

#include <AMReX_iMultiFab.H>

namespace ExaEpi
{
namespace Initialization
{
    void read_workerflow (const DemographicData& demo,
                          const TestParams& params,
                          const amrex::iMultiFab& unit_mf,
                          const amrex::iMultiFab& comm_mf,
                          AgentContainer& pc);
}
}

#endif
