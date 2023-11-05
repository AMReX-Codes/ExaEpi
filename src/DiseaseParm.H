#ifndef DISEASE_PARM_H_
#define DISEASE_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct DiseaseParm
{
    // Default contact matrix scaling factors
    amrex::Real pSC = amrex::Real(0.2);    // school
    amrex::Real pHSC = amrex::Real(0.3);    // high school
    amrex::Real pMSC = amrex::Real(0.3);    // middle school
    amrex::Real pESC = amrex::Real(0.3);    // elementary school
    amrex::Real pCO = amrex::Real(1.45);   // community
    amrex::Real pNH = amrex::Real(1.45);   // neighborhood
    amrex::Real pWO = amrex::Real(0.5);    // workplace
    amrex::Real pHC = amrex::Real(0.5);
    amrex::Real pHCC0 = amrex::Real(0.5);
    amrex::Real pHCC1 = amrex::Real(0.5);
    amrex::Real pFA =  amrex::Real(1.0);   // family
    amrex::Real pBAR = amrex::Real(-1.0);  // bar or other large social gatherings (<0 turns this off)

    int nstrain = 2;
    amrex::Real p_trans[2] = {amrex::Real(0.20), amrex::Real(0.30)};     // probability of transimission given contact
    amrex::Real p_asymp[2] = {amrex::Real(0.40), amrex::Real(0.40)};     // fraction of cases that are asymptomatic
    amrex::Real reduced_inf[2] = {amrex::Real(0.75), amrex::Real(0.75)}; // relative infectiousness of asymptomatic individuals

    // these are computed from the above
    amrex::Real xmit_comm[5] = {amrex::Real(.0000125*pCO), amrex::Real(.0000375*pCO), amrex::Real(.00010*pCO), amrex::Real(.00010*pCO), amrex::Real(.00015*pCO)};
    amrex::Real xmit_hood[5] = {amrex::Real(.00005*pNH), amrex::Real(.00015*pNH), amrex::Real(.00040*pNH), amrex::Real(.00040*pNH), amrex::Real(.00060*pNH)};
    amrex::Real xmit_nc_adult[5] = {amrex::Real(.08*pHC),  amrex::Real(.08*pHC), amrex::Real(.1*pHC), amrex::Real(.1*pHC), amrex::Real(.1*pHC)};
    amrex::Real xmit_nc_child[5] = {amrex::Real(.15*pHCC0), amrex::Real(.15*pHCC1), amrex::Real(.08*pHC), amrex::Real(.08*pHC), amrex::Real(.08*pHC)};
    amrex::Real xmit_work = amrex::Real(0.115*pWO);
    amrex::Real xmit_school[7] = {amrex::Real(0.), amrex::Real(.105*pHSC), amrex::Real(.125*pMSC), amrex::Real(.145*pESC), amrex::Real(.145*pESC), amrex::Real(.15), amrex::Real(.35)};

    /* Within family, from child or adult, for 5 age groups (here Thai paper, US:0.6->0.8)*/
    amrex::Real xmit_adult[5] = {amrex::Real(.3*pFA), amrex::Real(.3*pFA), amrex::Real(.4*pFA), amrex::Real(.4*pFA), amrex::Real(.4*pFA)};
    amrex::Real xmit_child[5] = {amrex::Real(.6*pFA), amrex::Real(.6*pFA), amrex::Real(.3*pFA), amrex::Real(.3*pFA), amrex::Real(.3*pFA)};

    amrex::Real xmit_child_SC[5], xmit_adult_SC[5], xmit_nc_child_SC[5], xmit_nc_adult_SC[5];
    amrex::Real xmit_comm_SC[5], xmit_hood_SC[5], xmit_sch_c2a[5], xmit_sch_a2c[5];

    amrex::Real infect = amrex::Real(1.0);

    amrex::Real reinfect_prob = amrex::Real(0.0);

    amrex::Real vac_eff = amrex::Real(1.0);

    amrex::Real Child_compliance, Child_HH_closure;

    void Initialize ();
};
#endif
