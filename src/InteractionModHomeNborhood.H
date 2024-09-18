/*! @file InteractionModHomeNborhood.H
 * \brief Contains the class describing agent interactions at neighborhood
 */

#ifndef _INTERACTION_MOD_HOME_NBORHOOD_H_
#define _INTERACTION_MOD_HOME_NBORHOOD_H_

#include <math.h>

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

template <typename PTDType>
struct HomeNborhoodCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        // this is the only case where we allow random travelers to interact
        return !isHospitalized(idx, ptd) && !ptd.m_idata[IntIdx::withdrawn][idx];
    }
};



/*! \brief Class describing agent interactions in the neighborhood/community */
template <typename PCType, typename PTDType, typename PType>
class InteractionModHomeNborhood : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModHomeNborhood () : InteractionModel<PCType, PTDType, PType>() {}

        /*! \brief default destructor */
        virtual ~InteractionModHomeNborhood () = default;

        /*! \brief Simulate agent interaction in the neighborhood/community */
        virtual void interactAgents (PCType& agents, MultiFab&) override;
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModHomeNborhood<PCType, PTDType, PType>::interactAgents (PCType& agents, MultiFab&) {
    BL_PROFILE("InteractionModHomeNborhood::interactAgents");
    int n_disease = agents.numDiseases();

    HomeNborhoodCandidate<PTDType> isCandidate;

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();

            GetCommunityIndex<PTDType> getCommunityIndex(agents.Geom(lev), mfi.validbox());

            int max_communities = getCommunityIndex.max();
            int max_nborhood = agents.getMaxGroup(IntIdx::nborhood) + 1;
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
                        auto nborhood = nborhood_ptr[i];
                        Gpu::Atomic::Add(&infected_community_d_ptr[community], 1);
                        Gpu::Atomic::Add(&infected_nborhood_d_ptr[community * max_nborhood + nborhood], 1);
                    }
                });
                Gpu::synchronize();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                        auto community = getCommunityIndex(ptd, i);
                        auto nborhood = nborhood_ptr[i];
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
