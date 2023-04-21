#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include "AgentContainer.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "Utils.H"

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

    void setInitialCases (AgentContainer& pc, const amrex::iMultiFab& unit_mf,
                          const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& comm_mf,
                          const CaseData& cases, const DemographicData& demo);

}
}

#endif
