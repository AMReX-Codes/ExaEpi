/*! @file InteractionModWorkNborhood.H
 * \brief Contains the class describing agent interactions at neighborhood
 */

#ifndef _INTERACTION_MOD_WORK_NBORHOOD_H_
#define _INTERACTION_MOD_WORK_NBORHOOD_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

template <typename PTDType>
struct WorkNborhoodCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) && !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};


/*! \brief Class describing agent interactions in the neighborhood/community */
template <typename PCType, typename PTDType, typename PType>
class InteractionModWorkNborhood : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModWorkNborhood () : InteractionModel<PCType, PTDType, PType>() {}

        /*! \brief default destructor */
        virtual ~InteractionModWorkNborhood() = default;

        /*! \brief Simulate agent interaction in the neighborhood/community */
        virtual void interactAgents (PCType& agents, MultiFab&) override;
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModWorkNborhood<PCType, PTDType, PType>::interactAgents (PCType& agents, MultiFab&) {
    BL_PROFILE("InteractionModWorkNborhood::interactAgents");
    int n_disease = agents.numDiseases();

    WorkNborhoodCandidate<PTDType> isCandidate;

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();
            auto home_nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();

            GetCommunityIndex<PTDType> getCommunityIndex(agents.Geom(lev), mfi.validbox());

            int max_communities = getCommunityIndex.max();
            int max_work_nborhood = agents.getMaxGroup(IntIdx::work_nborhood) + 1;
            int max_home_nborhood = agents.getMaxGroup(IntIdx::nborhood) + 1;
            int max_nborhood = max(max_work_nborhood, max_home_nborhood);
            AMREX_ALWAYS_ASSERT(max_nborhood <= np);

            for (int d = 0; d < n_disease; d++) {
                Gpu::DeviceVector<int> infected_community_d(max_communities, 0);
                Gpu::DeviceVector<int> infected_nborhood_d(max_communities * max_nborhood, 0);
                auto infected_community_d_ptr = infected_community_d.data();
                auto infected_nborhood_d_ptr = infected_nborhood_d.data();

                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = agents.getDiseaseParameters_d(d);
                auto lparm_h = agents.getDiseaseParameters_h(d);
                Real scale = 1.0_prt;  // TODO this should vary based on cell
                Real infect = lparm_h->infect * (1.0_rt - lparm_h->vac_eff);

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isInfectious(i, ptd, d) && isCandidate(i, ptd)) {
                        auto community = getCommunityIndex(ptd, i);
                        int nborhood = age_group_ptr[i] > 1 ? work_nborhood_ptr[i] : home_nborhood_ptr[i];
                        Gpu::Atomic::Add(&infected_community_d_ptr[community], 1);
                        Gpu::Atomic::Add(&infected_nborhood_d_ptr[community * max_nborhood + nborhood], 1);
                    }
                });
                Gpu::synchronize();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                        auto community = getCommunityIndex(ptd, i);
                        int nborhood = age_group_ptr[i] > 1 ? work_nborhood_ptr[i] : home_nborhood_ptr[i];
                        int num_infected_nborhood = infected_nborhood_d_ptr[community * max_nborhood + nborhood];
                        int num_infected_community = infected_community_d_ptr[community];
                        AMREX_ALWAYS_ASSERT(num_infected_community >= num_infected_nborhood);
                        ParticleReal comm_prob = 1.0_prt - infect * lparm->xmit_comm[ptd.m_idata[IntIdx::age_group][i]] * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(comm_prob, num_infected_community - num_infected_nborhood));
                        ParticleReal nborhood_prob = 1.0_prt - infect * lparm->xmit_hood[ptd.m_idata[IntIdx::age_group][i]] * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(nborhood_prob, num_infected_nborhood));
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}


#endif
