/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

template <typename PTDType>
struct SchoolCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) &&
                ptd.m_idata[IntIdx::school][idx] > 0 &&
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
        InteractionModSchool () : InteractionModel<PCType, PTDType, PType>() {}

        /*! \brief default destructor */
        virtual ~InteractionModSchool () = default;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents (PCType& agents, MultiFab&) override;
};


template <typename PCType, typename PTDType, typename PType>
void InteractionModSchool<PCType, PTDType, PType>::interactAgents (PCType& agents, MultiFab&) {
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
            auto school_ptr = soa.GetIntData(IntIdx::school).data();
            auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();

            GetCommunityIndex<PTDType> getCommunityIndex(agents.Geom(lev), mfi.validbox());

            int max_communities = getCommunityIndex.max();
            int max_school = agents.getMaxGroup(IntIdx::school) + 1;
            int max_nborhood = agents.getMaxGroup(IntIdx::nborhood) + 1;
            AMREX_ALWAYS_ASSERT(max_school <= 6);


            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    Gpu::DeviceVector<int> infected_school_d(max_communities * max_school, 0);
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
                            if (school_ptr[i] < 5) {
                                Gpu::Atomic::Add(&infected_school_d_ptr[community * max_school + school_ptr[i]], 1);
                            } else if (school_ptr[i] == 5) {
                                Gpu::Atomic::Add(&infected_daycare_d_ptr[community * max_nborhood + nborhood_ptr[i]], 1);
                            }
                        }
                    });
                    Gpu::synchronize();

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                            auto community = getCommunityIndex(ptd, i);
                            if (school_ptr[i] < 5) {
                                int num_infected_school = infected_school_d_ptr[community * max_school + school_ptr[i]];
                                ParticleReal school_prob = 0;
                                if (adults) {   // transmitters are adults
                                    if (age_group_ptr[i] <= 1) {  // Adult teacher/staff -> child student
                                        school_prob = 1.0_prt - infect * lparm->xmit_sch_a2c[school_ptr[i]] * scale; // * 0.25;
                                    } else {  // adult to adult
                                        school_prob = 1.0_prt - infect * lparm->xmit_school[school_ptr[i]] * scale;
                                    }
                                } else { // Transmitters are children
                                    if (age_group_ptr[i] <= 1) {  // Receiver j is a child
                                        school_prob = 1.0_prt - infect * lparm->xmit_school[school_ptr[i]] * scale;
                                    } else {  // Child student -> adult teacher/staff transmission
                                        school_prob = 1.0_prt - infect * lparm->xmit_sch_c2a[school_ptr[i]] * scale; // * 0.5;
                                    }
                                }
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(school_prob, num_infected_school));
                            } else if (school_ptr[i] == 5) {
                                int num_infected_daycare = infected_daycare_d_ptr[community * max_nborhood + nborhood_ptr[i]];
                                ParticleReal daycare_prob = 1.0_prt - infect * lparm->xmit_school[school_ptr[i]] * scale;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(daycare_prob, num_infected_daycare));
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
