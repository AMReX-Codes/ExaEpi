/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


#ifndef FAST_INTERACTIONS
/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at school. */
template <typename PTDType>
struct BinaryInteractionSchool {
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    ParticleReal operator() (const int infectious_i, /*!< Index of infectious agent */
                             const int susceptible_i, /*!< Index of susceptible agent */
                             const PTDType& a_ptd, /*!< Particle tile data */
                             const DiseaseParm* const a_lparm, /*!< disease paramters */
                             const Real a_social_scale  /*!< Social scale */) const noexcept {
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::school_id][infectious_i] == a_ptd.m_idata[IntIdx::school_id][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::school_grade][infectious_i] == a_ptd.m_idata[IntIdx::school_grade][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);

        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
        auto school_id_ptr = a_ptd.m_idata[IntIdx::school_id];

        // binned so that infectious and susceptible are in the same school
        int school_id = school_id_ptr[susceptible_i];
        //infect *= i_mask;
        //infect *= j_mask;
        // Elementary/middle/high school in common
        if (school_id > 0 && school_id < 5) {
            if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
                if (age_group_ptr[susceptible_i] <= 1) {  // Receiver j is a child
                    return a_lparm->xmit_school[school_id] * a_social_scale;
                } else {  // Child student -> adult teacher/staff transmission
                    return a_lparm->xmit_sch_c2a[school_id] * a_social_scale;// * 0.5; // hack for testing
                }
            } else {   // transmitter is an adult
                if (age_group_ptr[susceptible_i] <= 1) {  // Adult teacher/staff -> child student
                    return a_lparm->xmit_sch_a2c[school_id] * a_social_scale;// * 0.25; // hack for testing
                } else {  // adult to adult
                    return a_lparm->xmit_school[school_id] * a_social_scale;
                }
            }

        } else {
            if (nborhood_ptr[infectious_i] == nborhood_ptr[susceptible_i]) {
                if (school_id > 5) {             // Playgroup - currently not implemented in the initialization
                    return a_lparm->xmit_school[6] * a_social_scale;
                } else if (school_id == 5) {     // Day care
                    return a_lparm->xmit_school[5] * a_social_scale;
                }
            }
        }
        return 0.0_prt;
    }
};
#endif

template <typename PTDType>
struct SchoolCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) &&
                ptd.m_idata[IntIdx::school_id][idx] > 0 &&
                ptd.m_idata[IntIdx::school_closed][idx] == 0 &&
                !ptd.m_idata[IntIdx::withdrawn][idx] &&
                ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};



/*! \brief Class describing agent interactions at school */
template <typename PCType, typename PTDType, typename PType>
class InteractionModSchool : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModSchool (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModSchool () = default;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
#ifdef FAST_INTERACTIONS
            fastInteractSchool(agents);
#else
            interactAgentsImpl<InteractionModSchool<PCType, PTDType, PType>, PCType, PTDType,
                               SchoolCandidate<PTDType>,
                               BinaryInteractionSchool<PTDType>>(*this, agents, IntIdx::school_id);
#endif
        }

        void fastInteractSchool (PCType &agents);
};


template <typename PCType, typename PTDType, typename PType>
void InteractionModSchool<PCType, PTDType, PType>::fastInteractSchool (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    SchoolCandidate<PTDType> isCandidate;

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();

            GetCommunityIndex<PTDType> getCommunityIndex(agents.Geom(lev), mfi.validbox());

            int max_communities = getCommunityIndex.max();
            int max_school_grade = agents.getMaxGroup(IntIdx::school_grade) + 1;
            int max_school_id = agents.getMaxGroup(IntIdx::school_id) + 1;
            int max_nborhood = agents.getMaxGroup(IntIdx::nborhood) + 1;

            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    Gpu::DeviceVector<int> infected_school_d(max_communities * max_school_grade, 0);
                    auto infected_school_d_ptr = infected_school_d.data();
                    Gpu::DeviceVector<int> infected_daycare_d(max_communities * max_nborhood, 0);
                    auto infected_daycare_d_ptr = infected_daycare_d.data();

                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_prt;  // TODO this should vary based on cell
                    Real infect = lparm_h->infect * (1.0_rt - lparm_h->vac_eff);

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isCandidate(i, ptd) && isAnAdult(i, ptd) == adults) {
                            auto community = getCommunityIndex(ptd, i);
                            if (school_grade_ptr[i] <= 1) {
                                Gpu::Atomic::Add(&infected_daycare_d_ptr[community * max_nborhood + nborhood_ptr[i]], 1);
                            } else {
                                Gpu::Atomic::Add(&infected_school_d_ptr[community * max_school_id + school_id_ptr[i]], 1);
                            }
                        }
                    });
                    Gpu::synchronize();

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                            auto community = getCommunityIndex(ptd, i);
                            if (school_grade_ptr[i] <= 1) {
                                int num_infected_daycare = infected_daycare_d_ptr[community * max_nborhood + nborhood_ptr[i]];
                                ParticleReal daycare_prob = 1.0_prt - infect * lparm->xmit_daycare * scale;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(daycare_prob, num_infected_daycare));
                            } else {
                                int num_infected_school = infected_school_d_ptr[community * max_school_id + school_id_ptr[i]];
                                ParticleReal school_prob = 0;
                                if (adults) {   // transmitters are adults
                                    if (age_group_ptr[i] <= 1) {  // Adult teacher/staff -> child student
                                        Real xmit = lparm->xmit_elem_school_a2c;
                                        if (school_id_ptr[i] == 1) xmit = lparm->xmit_high_school_a2c;
                                        if (school_id_ptr[i] == 2) xmit = lparm->xmit_middle_school_a2c;
                                        school_prob = 1.0_prt - infect * xmit * scale; // * 0.25;
                                    } else {  // adult to adult - use high school transmission value
                                        Real xmit = lparm->xmit_elem_school;
                                        if (school_id_ptr[i] == 1) xmit = lparm->xmit_high_school;
                                        if (school_id_ptr[i] == 2) xmit = lparm->xmit_middle_school;
                                        school_prob = 1.0_prt - infect * xmit * scale;
                                    }
                                } else { // Transmitters are children
                                    if (age_group_ptr[i] <= 1) {  // Receiver j is a child
                                        Real xmit = lparm->xmit_elem_school;
                                        if (school_id_ptr[i] == 1) xmit = lparm->xmit_high_school;
                                        if (school_id_ptr[i] == 2) xmit = lparm->xmit_middle_school;
                                        school_prob = 1.0_prt - infect * xmit * scale;
                                    } else {  // Child student -> adult teacher/staff transmission
                                        Real xmit = lparm->xmit_elem_school_c2a;
                                        if (school_id_ptr[i] == 1) xmit = lparm->xmit_high_school_c2a;
                                        if (school_id_ptr[i] == 2) xmit = lparm->xmit_middle_school_c2a;
                                        school_prob = 1.0_prt - infect * xmit * scale; // * 0.5;
                                    }
                                }
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(school_prob, num_infected_school));
                            }
                        }
                    });
                    Gpu::synchronize();
                }
            }
        }
    }
}


#endif
