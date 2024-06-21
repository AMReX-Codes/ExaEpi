/*! @file Initialization.H
    \brief Defines #ExaEpi::Initialization namespace
*/

#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include "AgentContainer.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "Utils.H"

#include <AMReX_iMultiFab.H>

namespace ExaEpi
{
/*! \brief Namespace with initialization-related functions. */
namespace Initialization
{
    void read_workerflow (const DemographicData& demo,
                          const TestParams& params,
                          const amrex::iMultiFab& unit_mf,
                          const amrex::iMultiFab& comm_mf,
                          AgentContainer& pc,
                          amrex::iMultiFab& worker_counts);

    void setInitialCasesFromFile (AgentContainer& pc, const amrex::iMultiFab& unit_mf,
                                  const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& comm_mf,
                                  const CaseData& cases, const DemographicData& demo);

    void setInitialCasesRandom (AgentContainer& pc, const amrex::iMultiFab& unit_mf,
                                const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& comm_mf,
                                int num_cases, const DemographicData& demo);
}
}

#endif
