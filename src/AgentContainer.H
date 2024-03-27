/*! @file AgentContainer.H
    \brief Contains #AgentContainer class and related structs
*/
#ifndef AGENT_CONTAINER_H_
#define AGENT_CONTAINER_H_

#include <array>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particles.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Vector.H>

#include "DemographicData.H"
#include "DiseaseParm.H"

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
        incubation_period,      /*!< Time until infectious */
        infectious_period,      /*!< Length of time infectious */
        symptomdev_period,      /*!< Time until symptoms would develop */
        nattribs                /*!< number of real-type attribute*/
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

/*! \brief Assigns school by taking a random number between 0 and 100, and using
 *  default distribution to choose elementary/middle/high school. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int assign_school (const int nborhood, const amrex::RandomEngine& engine) {
    int il4 = amrex::Random_int(100, engine);
    int school = -1;

    if (il4 < 36) {
        school = 3 + (nborhood / 2);  /* elementary school */
    }
    else if (il4 < 68) {
        school = 2;  /* middle school */
    }

    else if (il4 < 93) {
        school = 1;  /* high school */
    }
    else {
        school = 0;  /* not in school, presumably 18-year-olds or some home-schooled */
    }
    return school;
}

/*! \brief Derived class from ParticleContainer that defines agents and their functions */
class AgentContainer
    : public amrex::ParticleContainer<0, 0, RealIdx::nattribs, IntIdx::nattribs>
{

public:

    /*! Constructor:
     *  + Initializes particle container for agents
     *  + Read in contact probabilities from command line input file
     *  + Read in disease parameters from command line input file
    */
    AgentContainer (const amrex::Geometry            & a_geom,  /*!< Physical domain */
                    const amrex::DistributionMapping & a_dmap,  /*!< Distribution mapping */
                    const amrex::BoxArray            & a_ba     /*!< Box array */ )
        : amrex::ParticleContainer<0, 0, RealIdx::nattribs, IntIdx::nattribs>(a_geom, a_dmap, a_ba)
    {
        h_parm = new DiseaseParm{};
        d_parm = (DiseaseParm*)amrex::The_Arena()->alloc(sizeof(DiseaseParm));

        {
            amrex::ParmParse pp("agent");
            pp.query("symptomatic_withdraw", m_symptomatic_withdraw);
        }

        {
            amrex::ParmParse pp("contact");
            pp.query("pSC", h_parm->pSC);
            pp.query("pCO", h_parm->pCO);
            pp.query("pNH", h_parm->pNH);
            pp.query("pWO", h_parm->pWO);
            pp.query("pFA", h_parm->pFA);
            pp.query("pBAR", h_parm->pBAR);
        }

        {
            amrex::ParmParse pp("disease");
            pp.query("nstrain", h_parm->nstrain);
            pp.query("reinfect_prob", h_parm->reinfect_prob);

            amrex::Vector<amrex::Real> p_trans(h_parm->nstrain);
            amrex::Vector<amrex::Real> p_asymp(h_parm->nstrain);
            amrex::Vector<amrex::Real> reduced_inf(h_parm->nstrain);

            pp.queryarr("p_trans", p_trans, 0, h_parm->nstrain);
            pp.queryarr("p_asymp", p_asymp, 0, h_parm->nstrain);
            pp.queryarr("reduced_inf", reduced_inf, 0, h_parm->nstrain);

            pp.query("vac_eff", h_parm->vac_eff);

            for (int i = 0; i < h_parm->nstrain; ++i) {
                h_parm->p_trans[i] = p_trans[i];
                h_parm->p_asymp[i] = p_asymp[i];
                h_parm->reduced_inf[i] = reduced_inf[i];
            }

            pp.query("incubation_length_mean", h_parm->incubation_length_mean);
            pp.query("infectious_length_mean", h_parm->infectious_length_mean);
            pp.query("symptomdev_length_mean", h_parm->symptomdev_length_mean);

            pp.query("incubation_length_std", h_parm->incubation_length_std);
            pp.query("infectious_length_std", h_parm->infectious_length_std);
            pp.query("symptomdev_length_std", h_parm->symptomdev_length_std);
        }

        h_parm->Initialize();
#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(d_parm, h_parm, sizeof(DiseaseParm));
#else
        std::memcpy(d_parm, h_parm, sizeof(DiseaseParm));
#endif
    }

    void initAgentsDemo (amrex::iMultiFab& /*num_residents*/,
                         amrex::iMultiFab& /*unit_mf*/,
                         amrex::iMultiFab& /*FIPS_mf*/,
                         amrex::iMultiFab& /*comm_mf*/,
                         DemographicData& /*demo*/);

    void initAgentsCensus (amrex::iMultiFab& num_residents,
                           amrex::iMultiFab& unit_mf,
                           amrex::iMultiFab& FIPS_mf,
                           amrex::iMultiFab& comm_mf,
                           DemographicData& demo);

    void moveAgentsRandomWalk ();

    void moveAgentsToWork ();

    void moveAgentsToHome ();

    void moveRandomTravel ();

    void updateStatus (amrex::MultiFab& ds);

    void interactAgents ();

    void interactAgentsHomeWork (amrex::MultiFab& mask, bool home);

    void infectAgents ();

    void shelterStart ();

    void shelterStop ();

    void generateCellData (amrex::MultiFab& mf) const;

    std::array<amrex::Long, 5> getTotals ();

    const DiseaseParm * getDiseaseParameters_h () const {
        return h_parm;
    }

    const DiseaseParm * getDiseaseParameters_d () const {
        return d_parm;
    }

protected:

    int m_symptomatic_withdraw = 0;

    DiseaseParm* h_parm;    /*!< Disease parameters */
    DiseaseParm* d_parm;    /*!< Disease parameters (GPU device) */

    /*! Map of home bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > m_bins_home;
    /*! Map of work bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > m_bins_work;
};

using AgentIterator = typename AgentContainer::ParIterType;

#endif
