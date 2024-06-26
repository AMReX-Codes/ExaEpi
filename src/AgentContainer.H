
/*! @file AgentContainer.H
    \brief Contains #AgentContainer class and related structs
*/
#ifndef AGENT_CONTAINER_H_
#define AGENT_CONTAINER_H_

#include <vector>
#include <string>
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

    using MFPtr = std::unique_ptr<MultiFab>;
    using MFPtrVec = std::vector<MFPtr>;

public:

    /*! Constructor:
     *  + Initializes particle container for agents
     *  + Read in contact probabilities from command line input file
     *  + Read in disease parameters from command line input file
    */
    AgentContainer (const amrex::Geometry            & a_geom,  /*!< Physical domain */
                    const amrex::DistributionMapping & a_dmap,  /*!< Distribution mapping */
                    const amrex::BoxArray            & a_ba,    /*!< Box array */
                    const int                        & a_num_diseases, /*!< Number of diseases */
                    const std::vector<std::string>   & a_disease_names /*!< names of the diseases */)
        : amrex::ParticleContainer< 0,
                                    0,
                                    RealIdx::nattribs,
                                    IntIdx::nattribs> (a_geom, a_dmap, a_ba),
          student_counts(a_ba, a_dmap, 5, 0),
          teacher_counts(a_ba, a_dmap, 6, 0)
    {
        BL_PROFILE("AgentContainer::AgentContainer");
        AMREX_ASSERT(m_num_diseases < ExaEpi::max_num_diseases);
        m_num_diseases = a_num_diseases;
        m_disease_names = a_disease_names;

        student_counts.setVal(0);  // Initialize the MultiFab to zero
        teacher_counts.setVal(0);

        add_attributes();

        {
            amrex::ParmParse pp("agent");
            pp.query("symptomatic_withdraw", m_symptomatic_withdraw);
            pp.query("shelter_compliance", m_shelter_compliance);
            pp.query("symptomatic_withdraw_compliance", m_symptomatic_withdraw_compliance);
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

        h_parm.resize(m_num_diseases);
        d_parm.resize(m_num_diseases);

        for (int d = 0; d < m_num_diseases; d++) {
            h_parm[d] = new DiseaseParm{};
            d_parm[d] = (DiseaseParm*)amrex::The_Arena()->alloc(sizeof(DiseaseParm));

            {
                std::string key = "contact";
                amrex::ParmParse pp(key);
                pp.query("pSC", h_parm[d]->pSC);
                pp.query("pCO", h_parm[d]->pCO);
                pp.query("pNH", h_parm[d]->pNH);
                pp.query("pWO", h_parm[d]->pWO);
                pp.query("pFA", h_parm[d]->pFA);
                pp.query("pBAR", h_parm[d]->pBAR);
            }

            {
                amrex::ParmParse pp("disease");
                pp.query("nstrain", h_parm[d]->nstrain);
                AMREX_ASSERT(h_parm[d]->nstrain <= 2);
                pp.query("reinfect_prob", h_parm[d]->reinfect_prob);

                amrex::Vector<amrex::Real> p_trans(h_parm[d]->nstrain);
                amrex::Vector<amrex::Real> p_asymp(h_parm[d]->nstrain);
                amrex::Vector<amrex::Real> reduced_inf(h_parm[d]->nstrain);

                // set correct default
                if (h_parm[d]->nstrain <= 2) {
                    for (int i = 0; i < h_parm[d]->nstrain; i++) {
                        p_trans[i] = h_parm[d]->p_trans[i];
                        p_asymp[i] = h_parm[d]->p_asymp[i];
                        reduced_inf[i] = h_parm[d]->reduced_inf[i];
                    }
                }

                pp.queryarr("p_trans", p_trans, 0, h_parm[d]->nstrain);
                pp.queryarr("p_asymp", p_asymp, 0, h_parm[d]->nstrain);
                pp.queryarr("reduced_inf", reduced_inf, 0, h_parm[d]->nstrain);

                pp.query("vac_eff", h_parm[d]->vac_eff);

                for (int i = 0; i < h_parm[d]->nstrain; ++i) {
                    h_parm[d]->p_trans[i] = p_trans[i];
                    h_parm[d]->p_asymp[i] = p_asymp[i];
                    h_parm[d]->reduced_inf[i] = reduced_inf[i];
                }

                pp.query("incubation_length_mean", h_parm[d]->incubation_length_mean);
                pp.query("infectious_length_mean", h_parm[d]->infectious_length_mean);
                pp.query("symptomdev_length_mean", h_parm[d]->symptomdev_length_mean);

                pp.query("incubation_length_std", h_parm[d]->incubation_length_std);
                pp.query("infectious_length_std", h_parm[d]->infectious_length_std);
                pp.query("symptomdev_length_std", h_parm[d]->symptomdev_length_std);

                pp.query("mean_immune_time", h_parm[d]->mean_immune_time);
                pp.query("immune_time_spread", h_parm[d]->immune_time_spread);
            }
            {
                std::string key = "disease_" + m_disease_names[d];
                amrex::ParmParse pp(key);
                pp.query("nstrain", h_parm[d]->nstrain);
                AMREX_ASSERT(h_parm[d]->nstrain <= 2);
                pp.query("reinfect_prob", h_parm[d]->reinfect_prob);

                amrex::Vector<amrex::Real> p_trans(h_parm[d]->nstrain);
                amrex::Vector<amrex::Real> p_asymp(h_parm[d]->nstrain);
                amrex::Vector<amrex::Real> reduced_inf(h_parm[d]->nstrain);

                // set correct default
                for (int i = 0; i < std::min(p_trans.size(),Long(2)); i++) {
                    p_trans[i] = h_parm[d]->p_trans[i];
                }
                for (int i = 0; i < std::min(p_asymp.size(),Long(2)); i++) {
                    p_asymp[i] = h_parm[d]->p_asymp[i];
                }
                for (int i = 0; i < std::min(reduced_inf.size(),Long(2)); i++) {
                    reduced_inf[i] = h_parm[d]->reduced_inf[i];
                }

                pp.queryarr("p_trans", p_trans, 0, h_parm[d]->nstrain);
                pp.queryarr("p_asymp", p_asymp, 0, h_parm[d]->nstrain);
                pp.queryarr("reduced_inf", reduced_inf, 0, h_parm[d]->nstrain);

                pp.query("vac_eff", h_parm[d]->vac_eff);

                for (int i = 0; i < h_parm[d]->nstrain; ++i) {
                    h_parm[d]->p_trans[i] = p_trans[i];
                    h_parm[d]->p_asymp[i] = p_asymp[i];
                    h_parm[d]->reduced_inf[i] = reduced_inf[i];
                }

                pp.query("incubation_length_mean", h_parm[d]->incubation_length_mean);
                pp.query("infectious_length_mean", h_parm[d]->infectious_length_mean);
                pp.query("symptomdev_length_mean", h_parm[d]->symptomdev_length_mean);

                pp.query("incubation_length_std", h_parm[d]->incubation_length_std);
                pp.query("infectious_length_std", h_parm[d]->infectious_length_std);
                pp.query("symptomdev_length_std", h_parm[d]->symptomdev_length_std);

                pp.query("mean_immune_time", h_parm[d]->mean_immune_time);
                pp.query("immune_time_spread", h_parm[d]->immune_time_spread);
            }

            h_parm[d]->Initialize();
#ifdef AMREX_USE_GPU
            amrex::Gpu::htod_memcpy(d_parm[d], h_parm[d], sizeof(DiseaseParm));
#else
            std::memcpy(d_parm[d], h_parm[d], sizeof(DiseaseParm));
#endif
        }
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

    void updateStatus (MFPtrVec& ds);

    void infectAgents ();

    void shelterStart ();

    void shelterStop ();

    void generateCellData (amrex::MultiFab& mf) const;

    std::array<amrex::Long, 9> getTotals (const int);

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
    inline const DiseaseParm* getDiseaseParameters_h (int d /*!< disease index */) const {
        return h_parm[d];
    }

    /*! \brief Return disease parameters object pointer (device) */
    inline const DiseaseParm* getDiseaseParameters_d (int d /*!< disease index */) const {
        return d_parm[d];
    }

    /*! \brief Return the number of diseases */
    inline int numDiseases() const {
        return m_num_diseases;
    }

    /* Since read_workerflow update worker status Unit-wise and not community-wise - Get total teacher needed for each unit */
    inline const amrex::Gpu::DeviceVector<int>& getUnitTotalTeacherCounts() const {
        return Unit_total_teacher_counts;
    }
    inline const amrex::Gpu::DeviceVector<int>& getUnitElem3TeacherCounts() const {
        return Unit_elem3_teacher_counts;
    }
    inline const amrex::Gpu::DeviceVector<int>& getUnitElem4TeacherCounts() const {
        return Unit_elem4_teacher_counts;
    }
    inline const amrex::Gpu::DeviceVector<int>& getUnitMidlTeacherCounts() const {
        return Unit_midl_teacher_counts;
    }
    inline const amrex::Gpu::DeviceVector<int>& getUnitHighTeacherCounts() const {
        return Unit_high_teacher_counts;
    }
    inline const amrex::Gpu::DeviceVector<int>& getUnitDaycrTeacherCounts() const {
        return Unit_daycr_teacher_counts;
    }

protected:

    amrex::iMultiFab student_counts;
    amrex::iMultiFab teacher_counts;
    amrex::Gpu::DeviceVector<int> Unit_total_teacher_counts;
    amrex::Gpu::DeviceVector<int> Unit_elem3_teacher_counts;
    amrex::Gpu::DeviceVector<int> Unit_elem4_teacher_counts;
    amrex::Gpu::DeviceVector<int> Unit_midl_teacher_counts;
    amrex::Gpu::DeviceVector<int> Unit_high_teacher_counts;
    amrex::Gpu::DeviceVector<int> Unit_daycr_teacher_counts;

    int m_num_diseases; /*!< Number of diseases */
    std::vector<std::string> m_disease_names; /*!< names of the diseases */

    int m_symptomatic_withdraw = 1;

    amrex::Real m_shelter_compliance = 0.95_rt;
    amrex::Real m_symptomatic_withdraw_compliance = 0.95_rt;

    std::vector<DiseaseParm*> h_parm;    /*!< Disease parameters */
    std::vector<DiseaseParm*> d_parm;    /*!< Disease parameters (GPU device) */

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

    /*! \brief Add runtime SoA attributes */
    void add_attributes();
};

using AgentIterator = typename AgentContainer::ParIterType;

#endif
