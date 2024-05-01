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
#include "InteractionModelLibrary.H"

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

    using PCType = AgentContainer;
    using PType = ParticleType;
    using PTileType = ParticleTileType;
    using PTDType = PTileType::ParticleTileDataType;
    using IntModel = InteractionModel<PCType,PTileType,PTDType,PType>;

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
            amrex::ParmParse pp("agent");
            pp.query("symptomatic_withdraw", m_symptomatic_withdraw);
            pp.query("shelter_compliance", m_shelter_compliance);
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
            using namespace ExaEpi;

            /* Create the interaction model objects and push to container */
            m_interactions.clear();
            m_interactions[InteractionNames::generic] = new InteractionModGeneric<PCType,PTileType,PTDType,PType>;
            m_interactions[InteractionNames::home] = new InteractionModHome<PCType,PTileType,PTDType,PType>;
            m_interactions[InteractionNames::work] = new InteractionModWork<PCType,PTileType,PTDType,PType>;
            m_interactions[InteractionNames::school] = new InteractionModSchool<PCType,PTileType,PTDType,PType>;
            m_interactions[InteractionNames::nborhood] = new InteractionModNborhood<PCType,PTileType,PTDType,PType>;
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

    void shelterStart ();

    void shelterStop ();

    void generateCellData (amrex::MultiFab& mf) const;

    std::array<amrex::Long, 9> getTotals ();

    void moveAgentsToWork ();

    void moveAgentsToHome ();

    /*! \brief Return bin pointer at a given mfi, tile and model name */
    inline amrex::DenseBins<PType>* getBins( const std::pair<int,int>& a_idx,
                                             const std::string& a_mod_name )
    {
        BL_PROFILE("AgentContainer::getBins");
        if (a_mod_name == ExaEpi::InteractionNames::home) {
            return &m_bins_home[a_idx];
        } else if (    (a_mod_name == ExaEpi::InteractionNames::work)
                    || (a_mod_name == ExaEpi::InteractionNames::school) ) {
            return &m_bins_work[a_idx];
        } else if (a_mod_name == ExaEpi::InteractionNames::nborhood) {
            if (m_at_work) { return &m_bins_work[a_idx]; }
            else           { return &m_bins_home[a_idx]; }
        } else {
            amrex::Abort("Invalid a_mod_name!");
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

    int m_symptomatic_withdraw = 0;

    amrex::Real m_shelter_compliance = 0.95_rt;

    DiseaseParm* h_parm;    /*!< Disease parameters */
    DiseaseParm* d_parm;    /*!< Disease parameters (GPU device) */

    /*! Map of home bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<PType> > m_bins_home;
    /*! Map of work bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<PType> > m_bins_work;

    std::map<std::string,IntModel*> m_interactions;

    /*! Flag to indicate if agents are at work */
    bool m_at_work;

    /*! \brief queries if a given interaction type (model) is available */
    inline bool haveInteractionModel( const std::string& a_mod_name ) const
    {
        BL_PROFILE("AgentContainer::haveInteractionModel");
        std::map<std::string,IntModel*>::const_iterator it(m_interactions.find(a_mod_name));
        return (it != m_interactions.end());
    }


};

using AgentIterator = typename AgentContainer::ParIterType;

#endif
