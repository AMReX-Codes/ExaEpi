/*! @file InitializeInfections.H
    \brief #Routines to implement the initial infections
*/

#pragma once

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>

#include <string>

#include "Utils.H"
#include "DemographicData.H"
#include "AgentContainer.H"
#include "CaseData.H"


void setInitialCasesFromFile (AgentContainer& pc,
                              const std::vector<CaseData>& cases,
                              const std::vector<std::string>& names,
                              const Vector<int> &FIPS_codes,
                              int ncomms,
                              const Vector<int> &unit_community_start,
                              iMultiFab &unit_mf,
                              iMultiFab &comm_mf,
                              const bool fast_bin);

void setInitialCasesRandom (AgentContainer& pc,
                            std::vector<int> num_cases,
                            const std::vector<std::string>& names,
                            int ncomms,
                            const Vector<int> &unit_community_start,
                            iMultiFab &unit_mf,
                            iMultiFab &comm_mf,
                            const bool fast_bin);



