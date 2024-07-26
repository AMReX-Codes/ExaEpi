/*! @file Initialization.H
    \brief Defines #ExaEpi::Initialization namespace
*/

#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include <vector>
#include <string>

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
                          AgentContainer& pc);

    void assignTeachersAndWorkgroup (const DemographicData& demo,
                          const amrex::iMultiFab& unit_mf,
                          AgentContainer& pc);

    void setInitialCasesFromFile (  AgentContainer& pc,
                                    const amrex::iMultiFab& unit_mf,
                                    const amrex::iMultiFab& FIPS_mf,
                                    const amrex::iMultiFab& comm_mf,
                                    const std::vector<CaseData>& cases,
                                    const std::vector<std::string>& names,
                                    const DemographicData& demo );

    void setInitialCasesRandom (    AgentContainer& pc,
                                    const amrex::iMultiFab& unit_mf,
                                    const amrex::iMultiFab& FIPS_mf,
                                    const amrex::iMultiFab& comm_mf,
                                    std::vector<int> num_cases,
                                    const std::vector<std::string>& names,
                                    const DemographicData& demo);
}
}

#endif
