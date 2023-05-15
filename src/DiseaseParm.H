#ifndef DISEASE_PARM_H_
#define DISEASE_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct DiseaseParm
{
    // Default contact matrix scaling factors
    amrex::Real pSC = amrex::Real(0.2);    // school
    amrex::Real pCO = amrex::Real(1.45);   // community
    amrex::Real pNH = amrex::Real(1.45);   // neighborhood
    amrex::Real pWO = amrex::Real(0.5);    // workplace
    amrex::Real pFA =  amrex::Real(1.0);   // family
    amrex::Real pBAR = amrex::Real(-1.0);  // bar or other large social gatherings (<0 turns this off)

    int nstrain = 2;
    amrex::Real p_trans[2] = {amrex::Real(0.20), amrex::Real(0.30)};     // probability of transimission given contact
    amrex::Real p_asymp[2] = {amrex::Real(0.40), amrex::Real(0.40)};     // fraction of cases that are asymptomatic
    amrex::Real reduced_inf[2] = {amrex::Real(0.75), amrex::Real(0.75)}; // relative infectiousness of asymptomatic individuals

    // these are computed from the above
    amrex::Real xmit_comm[5] = {amrex::Real(.0000125*pCO), amrex::Real(.0000375*pCO), amrex::Real(.00010*pCO), amrex::Real(.00010*pCO), amrex::Real(.00015*pCO)};
    amrex::Real xmit_work = amrex::Real(0.115*pWO);
    amrex::Real infect = amrex::Real(1.0);

    amrex::Real reinfect_prob = amrex::Real(0.0);

    void Initialize ();
};
#endif
