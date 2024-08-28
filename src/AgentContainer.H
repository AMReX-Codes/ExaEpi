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
#include "DiseaseStatus.H"
#include "HospitalModel.H"
#include "InteractionModelLibrary.H"
#include "AirTravelFlow.H"

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
          student_counts(a_ba, a_dmap, SchoolType::total_school_type, 0)
    {
        BL_PROFILE("AgentContainer::AgentContainer");

        m_num_diseases = a_num_diseases;
        AMREX_ASSERT(m_num_diseases < ExaEpi::max_num_diseases);
        m_disease_names = a_disease_names;

        student_counts.setVal(0);  // Initialize the MultiFab to zero

        add_attributes();

        {
            amrex::ParmParse pp("agent");
            pp.query("symptomatic_withdraw", m_symptomatic_withdraw);
            pp.query("shelter_compliance", m_shelter_compliance);
            pp.query("symptomatic_withdraw_compliance", m_symptomatic_withdraw_compliance);
            pp.queryarr("student_teacher_ratios", student_teacher_ratios);

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
            m_interactions[InteractionNames::random] = new InteractionModRandom<PCType,PTileType, PTDType, PType>;
            m_interactions[InteractionNames::airTravel] = new InteractionModAirTravel<PCType,PTileType, PTDType, PType>;

            m_hospital = std::make_unique<HospitalModel<PCType,PTileType,PTDType,PType>>();
        }

        h_parm.resize(m_num_diseases);
        d_parm.resize(m_num_diseases);

        for (int d = 0; d < m_num_diseases; d++) {
            h_parm[d] = new DiseaseParm{};
            d_parm[d] = (DiseaseParm*)amrex::The_Arena()->alloc(sizeof(DiseaseParm));

            h_parm[d]->readContact();
            // first read inputs common to all diseases
            h_parm[d]->readInputs("disease");
            // now read any disease-specific input, if available
            h_parm[d]->readInputs(std::string("disease_"+m_disease_names[d]));
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

    void interactRandomTravel(amrex::MultiFab&, AgentContainer& on_travel_pc);

    void interactAirTravel(amrex::MultiFab&, AgentContainer& on_travel_pc);

    void moveAgentsRandomWalk ();

    void moveRandomTravel (const amrex::iMultiFab& unit_mf);
    void moveAirTravel (const amrex::iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo);
    void setAirTravel (const iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo);

    void returnRandomTravel (const AgentContainer& on_travel_pc);

    void returnAirTravel (const AgentContainer& on_travel_pc);

    void updateStatus (MFPtrVec&);

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
        } else if (a_mod_name == ExaEpi::InteractionNames::random || a_mod_name == ExaEpi::InteractionNames::airTravel) {
            return &m_bins_random[a_idx];
        } else {
            amrex::Abort("Invalid a_mod_name!");
            return nullptr;
        }
    }

    /*! \brief Return flag indicating if agents are at work */
    inline bool isAtWork() const {
        return m_at_work;
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
    inline const amrex::Gpu::DeviceVector<int>& getUnitTeacherCounts() const {
        return unit_teacher_counts_d;
    }
    inline const amrex::Gpu::DeviceVector<int>& getCommDayCrTeacherCounts() const {
        return comm_teacher_counts_daycr_d;
    }
    inline const amrex::Gpu::DeviceVector<int>& getCommHighTeacherCounts() const {
        return comm_teacher_counts_high_d;
    }
    inline const amrex::Gpu::DeviceVector<int>& getCommMiddleTeacherCounts() const {
        return comm_teacher_counts_middle_d;
    }
    inline const amrex::Gpu::DeviceVector<int>& getCommElem3TeacherCounts() const {
        return comm_teacher_counts_elem3_d;
    }
    inline const amrex::Gpu::DeviceVector<int>& getCommElem4TeacherCounts() const {
        return comm_teacher_counts_elem4_d;
    }
    inline const amrex::Gpu::DeviceVector<int>& getCommTeacherCounts() const {
        return comm_teacher_counts_total_d;
    }

    /*! \brief return the symptomatic withdrawal status */
    inline int symptomaticWithdraw() const {
        return m_symptomatic_withdraw;
    }

    /*! \brief return the symptomatic withdrawal compliance */
    inline amrex::Real symptomaticWithdrawCompliance() const {
        return m_symptomatic_withdraw_compliance;
    }

protected:

    amrex::iMultiFab student_counts;
    amrex::Gpu::DeviceVector<int> unit_teacher_counts_d;
    amrex::Gpu::DeviceVector<int> comm_teacher_counts_total_d;
    amrex::Gpu::DeviceVector<int> comm_teacher_counts_high_d;
    amrex::Gpu::DeviceVector<int> comm_teacher_counts_middle_d;
    amrex::Gpu::DeviceVector<int> comm_teacher_counts_elem3_d;
    amrex::Gpu::DeviceVector<int> comm_teacher_counts_elem4_d;
    amrex::Gpu::DeviceVector<int> comm_teacher_counts_daycr_d;

    std::vector<long> student_teacher_ratios = {20, 20, 20, 20, 20, 1000000000}; // ignore daycare teachers since daycare interaction isnt handled yet!

    int m_num_diseases; /*!< Number of diseases */
    std::vector<std::string> m_disease_names; /*!< names of the diseases */

    int m_symptomatic_withdraw = 1; /*!< whether symptomatic agents withdraw or not */

    amrex::Real m_shelter_compliance = 0.95_rt; /*!< Shelter-in-place compliance rate */
    amrex::Real m_symptomatic_withdraw_compliance = 0.95_rt; /*!< Symptomatic withdrawal compliance rate */

    std::vector<DiseaseParm*> h_parm;    /*!< Disease parameters */
    std::vector<DiseaseParm*> d_parm;    /*!< Disease parameters (GPU device) */

    /*! Map of home bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<PType> > m_bins_home;
    /*! Map of work bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsHomeWork() */
    std::map<std::pair<int, int>, amrex::DenseBins<PType> > m_bins_work;
    /*! Map of random travel bins (of agents) indexed by MultiFab iterator and tile index;
        see AgentContainer::interactAgentsRandom() */
    std::map<std::pair<int, int>, amrex::DenseBins<PType> > m_bins_random;

    std::map<std::string,IntModel*> m_interactions; /*!< Map of interaction models */
    std::unique_ptr<HospitalModel<PCType,PTileType,PTDType,PType>> m_hospital; /*!< hospital model */

    /*! Flag to indicate if agents are at work */
    bool m_at_work;

    /*! Disease status update model */
    DiseaseStatus<PCType,PTileType,PTDType,PType> m_disease_status;

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
