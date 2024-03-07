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

#include "AgentDefinitions.H"
#include "DemographicData.H"
#include "DiseaseParm.H"
#include "InteractionLocations.H"

#if 0  // uncomment for EpiCast particle type

struct RealIdx
{
    enum {
#ifdef TWOSTRAIN
        prob1 = 0,    // daily probability of transmission of strain 1
        prob2,        // daily probability of transmission of strain 2
#else
        prob = 0,     // daily probability of transmission
#endif
#ifdef ATTRIBUTE
        /* Attribution of infection to particular mixing group source (other = community) */
        p_family,      // Household (family)
        p_school,      // School (children)
        p_work,        // Workplace
        p_nc,          // Neighborhood cluster
        p_hood,        // Neighborhood
        p_bar,         // Bars/restaurants
        p_school_mix,  // Student <--> teacher/staff transmission
        p_bc,          // Business-customer interaction
#endif
        nattribs
    };
};

struct IntIdx
{
    enum {
          type = 0,
          tag,

          /* home community (i,j,k) */
          comm_home_i,
          comm_home_j,
          comm_home_k,

          /* work (regular travel) community (i,j,k) */
          comm_work_i,
          comm_work_j,
          comm_work_k,

          /* irregular travel community (i,j,k) */
          comm_travel_i,
          comm_travel_j,
          comm_travel_k,

          /* timesteps left on travel (0 = not on travel) */
          trip_timer,  /* this could be an unsigned char (0-255) */

          /*
            bits 0/1: home neighborhood    ( nbor_all & 3 )
            bits 2/3: work neighborhood    ( (nbor_all >> 2) & 3 )
            bits 4/5: travel neighborhood  ( (nbor_all >> 4) & 3 )
          */
          nbor_all,

          /* neighborhood ID (lowest 2 bits only are used) */
          nborhood,          /* 0 - 255 (8 bits) */

          /* Workgroup tag - community-wide (unsigned char = 0-255, 0 = nonworker) */
          workgroup,

#ifdef USE_NAICS
          NAICS,  // Industry code for workplace
#endif

          /* Family id tag - valid only within a common nborhood */
          family,

          /*
            ATM - this is the original epicast bitpacking code, right now I just
            set 0 or 1 for infected or not.

            bits 0-2: age (groups 0-4)
            bits 3-7: disease length counter (number of days, 0-18)
            bits 8-10: disease status
            (0 = uninfected or recovered, 1 = incubation, 2 = prodromal, 3 = pox)
            bits 11-12: vaccination status (in order of increasing efficiency:
                    0 = none, 1 = old, 2 = new, 3 = recovered from outbreak)
                    for flu, antivirals are #2 (VACC2 mask above)
            bit 13: withdrawn?
            bit 14: diagnosed individual currently obeying isolation at home
            bit 15: prophylaxed contact of a diagnosed individual who is
                    currently obeying a quarantine
          */
          status,       /* 0 - 65535 (16 bits) */
          hh_size,      /* Household size (1-7) */
          vacc_tier,    // Tier for vaccination (0 = never receiving vaccine)
          vacc_timer,   // Days until full vaccine efficacy (0-255)

          /*
            timesteps left on vaccine/antiviral treatment
            0  : not treated
            -1 : ineligible for treatment

            For COVID-19, let's use this as follows:
            Hospitalized (mean 3 days from symptom onset, 4-10 day duration)
            ICU (15-20% of those hospitalized)
            Ventilated (30-50% of those in ICU)
            If hospitalized (decide on day 3 of symptoms):
            set the self-isolation bit (14) in pt->status, and
            set treatment_timer to 4-10 (depending on age group, just use mean values)
            decrement this each day, then SPaSM_delete_particle() when it's 0
            If ICU (decide when hospitalized), then also:
            set treatment_timer to 14-20 (depending on age group, just use mean values)
            decrement this each day, then when it's 10:
            mark 50% of age group 4 (65+) as dead before SPaSM_delete_particle()
            If ventilated (decide when hospitalized), then also:
            set treatment_timer to 24-30 (depending on age group, just use mean values)
            decrement this each day, then when it reaches 10:
            mark {15%, 25%, 5%, 60%, 100%} as dead before SPaSM_delete_particle()
          */
          treatment_timer,

          /*
            Playgroup/daycare/school -- note that playgroup/daycare id numbers
            (5+) are only valid within a common nborhood field, so both must
            be checked!
            0: not in school (adult or 7% of age group 1)
            1: community high school
            2: community middle school
            3: elementary school for neighborhoods 0 and 1
            4: elementary school for neighborhoods 2 and 3
            5: neighborhood daycare center (avg. 14 toddlers)
            6+: neighborhood playgroup (4 toddlers)
          */
          school,

#ifdef TWOSTRAIN
          /*
            Currently this is just a boolean, indicating whether this person is infected
            with the original strain (0) or second strain (1) - but other bits will surely
            be put to good use.
          */
          strain2,
#endif
          nattribs
    };
};

#endif // Epicast particle type

// simplified particle type

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
        BL_PROFILE("AgentContainer::AgentContainer");

        h_parm = new DiseaseParm{};
        d_parm = (DiseaseParm*)amrex::The_Arena()->alloc(sizeof(DiseaseParm));

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
            using namespace ExaEpi;

            /* Create the interaction location objects and push to container */
            m_interactions.clear();
            m_interactions[LocationNames::generic] = new InteractionLocGeneric<AgentContainer,AgentType>;
            m_interactions[LocationNames::home] = new InteractionLocHome<AgentContainer,AgentType>;
            m_interactions[LocationNames::work] = new InteractionLocWork<AgentContainer,AgentType>;
            m_interactions[LocationNames::school] = new InteractionLocSchool<AgentContainer,AgentType>;
            m_interactions[LocationNames::nborhood] = new InteractionLocNborhood<AgentContainer,AgentType>;
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

    void morningCommute(amrex::MultiFab&);

    void eveningCommute(amrex::MultiFab&);

    void interactDay(amrex::MultiFab&);

    void interactEvening(amrex::MultiFab&);

    void interactNight(amrex::MultiFab&);

    void moveAgentsRandomWalk ();

    void moveRandomTravel ();

    void updateStatus (amrex::MultiFab& ds);

    void infectAgents ();

    void generateCellData (amrex::MultiFab& mf) const;

    std::array<amrex::Long, 5> printTotals ();

    void moveAgentsToWork ();

    void moveAgentsToHome ();

    /*! \brief Return bin pointer at a given mfi, tile and location name */
    inline amrex::DenseBins<AgentContainer::ParticleType>* getBins( const std::pair<int,int>& a_idx,
                                                                    const std::string& a_loc_name )
    {
        BL_PROFILE("AgentContainer::getBins");
        if (a_loc_name == ExaEpi::LocationNames::home) {
            return &m_bins_home[a_idx];
        } else if (    (a_loc_name == ExaEpi::LocationNames::work)
                    || (a_loc_name == ExaEpi::LocationNames::school) ) {
            return &m_bins_work[a_idx];
        } else if (a_loc_name == ExaEpi::LocationNames::nborhood) {
            if (m_at_work) { return &m_bins_work[a_idx]; }
            else           { return &m_bins_home[a_idx]; }
        } else {
            amrex::Abort("Invalid a_loc_name!");
            return nullptr;
        }
    }

    /*! \brief Return disease parameters object pointer (host) */
    inline const DiseaseParm* getDiseaseParameters_h () const {
        return h_parm;
    }

    /*! \brief Return disease parameters object pointer (device) */
    inline const DiseaseParm* getDiseaseParameters_d () const {
        return d_parm;
    }

protected:

    using AgentType = ParticleType;

    DiseaseParm* h_parm;    /*!< Disease parameters */
    DiseaseParm* d_parm;    /*!< Disease parameters (GPU device) */

    /*! Map of home bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > m_bins_home;
    /*! Map of work bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > m_bins_work;

    std::map<std::string,InteractionLocation<AgentContainer,AgentType>*> m_interactions;

    /*! Flag to indicate if agents are at work */
    bool m_at_work;

    /*! \brief queries if a given interaction type (location) is available */
    inline bool haveInteractionLoc( const std::string& a_loc_name ) const
    {
        BL_PROFILE("AgentContainer::haveInteractionLoc");
        std::map<std::string,InteractionLocation<AgentContainer,AgentType>*>::const_iterator
            it(m_interactions.find(a_loc_name));
        return (it != m_interactions.end());
    }


};

using AgentIterator = typename AgentContainer::ParIterType;

#endif
