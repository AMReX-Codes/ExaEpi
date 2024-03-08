/*! @file AgentDefinitions.H
    \brief Contains some basic definitions for #AgentContainer
*/

#ifndef _AGENT_DEF_H_
#define _AGENT_DEF_H_

#include <AMReX_Particles.H>

/*! \brief Real-type SoA attributes of agent */
struct RealIdx
{
    enum {
        /* Disease counter starts after infection. */
        disease_counter = 0,    /*!< Counter since start of infection */
        treatment_timer,        /*!< Timer since hospital admission */
        prob,                   /*!< Probability of infection */
        incubation_period,      /*!< Time until infectious */
        infectious_period,      /*!< Length of time infectious */
        symptomdev_period,      /*!< Time until symptoms would develop */
        nattribs                /*!< number of real-type attribute*/
    };
};

/*! \brief Integer-type SoA attributes of agent */
struct IntIdx
{
    enum {
        status = 0,     /*!< Disease status (#Status) */
        strain,         /*!< virus strain */
        age_group,      /*!< Age group (under 5, 5-17, 18-29, 30-64, 65+) */
        family,         /*!< Family ID */
        home_i,         /*!< home location index */
        home_j          /*!< home location index */,
        work_i          /*!< work location index */,
        work_j          /*!< work location index */,
        nborhood,       /*!< home neighborhood ID */
        school,         /*!< school type (elementary, middle, high, none) */
        workgroup,      /*!< workgroup ID */
        work_nborhood,  /*!< work neighborhood ID */
        withdrawn,      /*!< quarantine status */
        symptomatic,    /*!< currently symptomatic? */
        nattribs        /*!< number of integer-type attribute */
    };
};

/*! \brief Disease status */
struct Status
{
    enum {
        never = 0,   /*!< never infected */
        infected,    /*!< infected */
        immune,      /*!< no longer infected, immune. lasts 6 months. */
        susceptible, /*!< no longer infected, no longer immnune */
        dead         /*!< passed away */
    };
};

/*! \brief Is an agent infected but not infectious? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool notInfectiousButInfected ( const int      a_idx, /*!< Agent index */
                                const PTDType& a_ptd  /*!< Particle tile data */ )
{
    return (    (a_ptd.m_idata[IntIdx::status][a_idx] == Status::infected)
             && (a_ptd.m_rdata[RealIdx::disease_counter][a_idx] < a_ptd.m_rdata[RealIdx::incubation_period][a_idx]) );
}

/*! \brief Is an agent infectious? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isInfectious ( const int      a_idx, /*!< Agent index */
                    const PTDType& a_ptd  /*!< Particle tile data */ )
{
    return (   (a_ptd.m_idata[IntIdx::status][a_idx] == Status::infected)
            && (a_ptd.m_rdata[RealIdx::disease_counter][a_idx] >= a_ptd.m_rdata[RealIdx::incubation_period][a_idx]) );
}

/*! \brief Is an agent susceptible? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isSusceptible ( const int      a_idx, /*!< Agent index */
                     const PTDType& a_ptd  /*!< Particle tile data */ )
{
    return (   (a_ptd.m_idata[IntIdx::status][a_idx] != Status::immune)
            && (a_ptd.m_idata[IntIdx::status][a_idx] != Status::dead)
            && (a_ptd.m_idata[IntIdx::status][a_idx] != Status::infected) );
}

/*! \brief Is an agent not susceptible (i.e., dead, immune, or already infected)? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool notSusceptible ( const int      a_idx, /*!< Agent index */
                      const PTDType& a_ptd  /*!< Particle tile data */ )
{
    return (   (a_ptd.m_idata[IntIdx::status][a_idx] == Status::immune)
            || (a_ptd.m_idata[IntIdx::status][a_idx] == Status::dead) );
}

#endif

