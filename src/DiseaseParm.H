/*! @file DiseaseParm.H
    \brief #DiseaseParm struct definition
*/

#ifndef DISEASE_PARM_H_
#define DISEASE_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

/*! \brief Disease parameters

    Structure containing disease parameters.
*/
struct DiseaseParm
{
    // Default contact matrix scaling factors
    amrex::Real pSC = amrex::Real(0.2);     /*!< Contact matrix scaling factor - school */
    amrex::Real pHSC = amrex::Real(0.3);    /*!< Contact matrix scaling factor - high school */
    amrex::Real pMSC = amrex::Real(0.3);    /*!< Contact matrix scaling factor - middle school */
    amrex::Real pESC = amrex::Real(0.3);    /*!< Contact matrix scaling factor - elementary school */
    amrex::Real pCO = amrex::Real(1.45);    /*!< Contact matrix scaling factor - community */
    amrex::Real pNH = amrex::Real(1.45);    /*!< Contact matrix scaling factor - neighborhood */
    amrex::Real pWO = amrex::Real(0.5);     /*!< Contact matrix scaling factor - workplace */
    amrex::Real pHC = amrex::Real(0.5);     /*!< Contact matrix scaling factor - ?? */
    amrex::Real pHCC0 = amrex::Real(0.5);   /*!< Contact matrix scaling factor - ?? */
    amrex::Real pHCC1 = amrex::Real(0.5);   /*!< Contact matrix scaling factor - ?? */
    amrex::Real pFA =  amrex::Real(1.0);    /*!< Contact matrix scaling factor - family */
    amrex::Real pBAR = amrex::Real(-1.0);   /*!< Contact matrix scaling factor - bar or other large social gatherings (<0 turns this off) */

    int nstrain = 2;  /*!< Number of strains */
    amrex::Real p_trans[2] = {amrex::Real(0.20), amrex::Real(0.30)};     /*!< probability of transimission given contact */
    amrex::Real p_asymp[2] = {amrex::Real(0.40), amrex::Real(0.40)};     /*!< fraction of cases that are asymptomatic */
    amrex::Real reduced_inf[2] = {amrex::Real(0.75), amrex::Real(0.75)}; /*!< relative infectiousness of asymptomatic individuals */

    // these are computed from the above
    /*! Community transmission probabilities for each age group (<5, 5-17, 18-29, 30-64, 65+) */
    amrex::Real xmit_comm[5] = {amrex::Real(.0000125*pCO), amrex::Real(.0000375*pCO), amrex::Real(.00010*pCO), amrex::Real(.00010*pCO), amrex::Real(.00015*pCO)};
    /*! Neighborhood transmission probabilities for each age group (<5, 5-17, 18-29, 30-64, 65+) */
    amrex::Real xmit_hood[5] = {amrex::Real(.00005*pNH), amrex::Real(.00015*pNH), amrex::Real(.00040*pNH), amrex::Real(.00040*pNH), amrex::Real(.00060*pNH)};
    /*! Transmission probabilities of adults for each age group (<5, 5-17, 18-29, 30-64, 65+) for neighborhood cluster */
    amrex::Real xmit_nc_adult[5] = {amrex::Real(.08*pHC),  amrex::Real(.08*pHC), amrex::Real(.1*pHC), amrex::Real(.1*pHC), amrex::Real(.1*pHC)};
    /*! Transmission probabilities of children for each age group (<5, 5-17, 18-29, 30-64, 65+) for neighborhood cluster */
    amrex::Real xmit_nc_child[5] = {amrex::Real(.15*pHCC0), amrex::Real(.15*pHCC1), amrex::Real(.08*pHC), amrex::Real(.08*pHC), amrex::Real(.08*pHC)};
    /*! Work transmission probabilities for each age group (<5, 5-17, 18-29, 30-64, 65+) */
    amrex::Real xmit_work = amrex::Real(0.115*pWO);
    /*! School transmission probabilities for each age group (<5, 5-17, 18-29, 30-64, 65+) */
    amrex::Real xmit_school[7] = {amrex::Real(0.), amrex::Real(.105*pHSC), amrex::Real(.125*pMSC), amrex::Real(.145*pESC), amrex::Real(.145*pESC), amrex::Real(.15), amrex::Real(.35)};

    /* Within family, from child or adult, for 5 age groups (here Thai paper, US:0.6->0.8)*/
    /*! Transmission probability within family for each age-group of adult */
    amrex::Real xmit_adult[5] = {amrex::Real(.3*pFA), amrex::Real(.3*pFA), amrex::Real(.4*pFA), amrex::Real(.4*pFA), amrex::Real(.4*pFA)};
    /*! Transmission probability within family for each age-group of child */
    amrex::Real xmit_child[5] = {amrex::Real(.6*pFA), amrex::Real(.6*pFA), amrex::Real(.3*pFA), amrex::Real(.3*pFA), amrex::Real(.3*pFA)};

    amrex::Real xmit_child_SC[5],     /*!< Transmission probability of child - school closed */
                xmit_adult_SC[5],     /*!< Transmission probability of adult - school closed */
                xmit_nc_child_SC[5],  /*!< Transmission probability for neighborhood cluster (child) - school closed ?? */
                xmit_nc_adult_SC[5];  /*!< Transmission probability for neighborhood cluster (adult) - school closed ?? */
    amrex::Real xmit_comm_SC[5],      /*!< Transmission probability in community - school closed */
                xmit_hood_SC[5],      /*!< Transmission probability in neighborhood - school closed ?? */
                xmit_sch_c2a[5],      /*!< Transmission probability in school - child to adult */
                xmit_sch_a2c[5];      /*!< Transmission probability in school - adult to child */

    amrex::Real infect = amrex::Real(1.0); /*!< Overall multiplier for agent's infectiousness */

    amrex::Real reinfect_prob = amrex::Real(0.0); /*!< Reinfection probability */

    amrex::Real vac_eff = amrex::Real(1.0); /*!< Vaccine efficacy */

    amrex::Real Child_compliance, /*!< Child compliance with masking ?? */
                Child_HH_closure; /*!< Multiplier for household contacts during school closure */

    int incubation_length  = 3;   /*!< Incubation period of disease (in days) */
    int infectious_length  = 6;   /*!< Number of days person is infectious */
    int symptomatic_length = 5;   /*!< Number of days person is symptomatic */ //note, this does not affect the model yet

    void Initialize ();
};
#endif
