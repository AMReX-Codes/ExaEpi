#ifndef DISEASE_PARM_H_
#define DISEASE_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct DiseaseParm
{
    // Default contact matrix scaling factors
    amrex::Real pSC = 0.2;    // school
    amrex::Real pCO = 1.45;   // community
    amrex::Real pNH = 1.45;   // neighborhood
    amrex::Real pWO = 0.5;    // workplace
    amrex::Real pFA =  1.;    // family
    amrex::Real pBAR = -1.;    // bar or other large social gatherings (<0 turns this off)

    int nstrain = 2;
    amrex::Real p_trans[2] = {0.20, 0.30};     // probability of transimission given contact
    amrex::Real p_asymp[2] = {0.40, 0.40};     // fraction of cases that are asymptomatic
    amrex::Real reduced_inf[2] = {0.75, 0.75}; // relative infectiousness of asymptomatic individuals

    // these are computed from the above
    amrex::Real xmit_comm[5] = {.0000125*pCO, .0000375*pCO, .00010*pCO, .00010*pCO, .00015*pCO};
    amrex::Real xmit_work = 0.115*pWO;
    amrex::Real infect = 1.0;

    amrex::Real reinfect_prob = 0.0;

    void Initialize ();
};

#endif
