/*! @file AgentDefinitions.H
    \brief Contains some basic definitions for #AgentContainer
*/

#ifndef _AGENT_DEF_H_
#define _AGENT_DEF_H_

/*! \brief Real-type SoA attributes of agent */
struct RealIdx
{
    enum {
        /*
          Disease counter starts after infection.
         */
        disease_counter = 0,    /*!< Counter since start of infection */
        treatment_timer,        /*!< Timer since hospital admission */
        prob,                   /*!< Probability of infection */
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
        withdrawn,      /*!< quarrantine status */
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

#endif

