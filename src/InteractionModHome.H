/*! @file InteractionModHome.H
 * \brief Contains the class describing agent interactions at home
 */

#ifndef _INTERACTION_MOD_HOME_H_
#define _INTERACTION_MOD_HOME_H_

#include "InteractionModel.H"
#include "DiseaseParm.H"
#include "AgentDefinitions.H"

using namespace amrex;

#define FAMILIES_PER_CLUSTER 4

template <typename PTDType>
struct HomeCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) && ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Class describing agent interactions at home */
template <typename PCType, typename PTDType, typename PType>
class InteractionModHome : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModHome() : InteractionModel<PCType, PTDType, PType>() {}

        /*! \brief default destructor */
        virtual ~InteractionModHome() = default;

        /*! \brief Simulate agent interaction at home */
        virtual void interactAgents (PCType& agents, MultiFab&) override;
};


template <typename PCType, typename PTDType, typename PType>
void InteractionModHome<PCType, PTDType, PType>::interactAgents (PCType& agents, MultiFab&) {
    BL_PROFILE("InteractionModHome::interactAgents");
    int n_disease = agents.numDiseases();

    HomeCandidate<PTDType> isHomeCandidate;

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto family_ptr = soa.GetIntData(IntIdx::family).data();
            auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();

            GetCommunityIndex<PTDType> getCommunityIndex(agents.Geom(lev), mfi.validbox());

            // calculate the max group values for indexing
            int max_communities = getCommunityIndex.max();
            int max_nborhood = agents.getMaxGroup(IntIdx::nborhood) + 1;
            int max_family = agents.getMaxGroup(IntIdx::family) + 1;
            int max_nc = max_family / FAMILIES_PER_CLUSTER + 1;

            for (int d = 0; d < n_disease; d++) {
                // calculate separately for infected children and infected adults, since they have different transmission rates
                for (auto adults : {true, false}) {
                    // set vectors to store counts of infected agents for each group
                    Gpu::DeviceVector<int> infected_family_d(max_communities * max_family, 0);
                    auto infected_family_d_ptr = infected_family_d.data();
                    Gpu::DeviceVector<int> infected_family_not_withdrawn_d(max_communities * max_family, 0);
                    auto infected_family_not_withdrawn_d_ptr = infected_family_not_withdrawn_d.data();
                    Gpu::DeviceVector<int> infected_nc_d(max_communities * max_nc * max_nborhood, 0);
                    auto infected_nc_d_ptr = infected_nc_d.data();

                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_prt;  // TODO this should vary based on cell
                    Real infect = lparm_h->infect * (1.0_rt - lparm_h->vac_eff);
                    // loop to count infectious agents in each group
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isHomeCandidate(i, ptd) && (isAnAdult(i, ptd) == adults)) {
                            auto community = getCommunityIndex(ptd, i);
                            Gpu::Atomic::Add(&infected_family_d_ptr[community * max_family + family_ptr[i]], 1);
                            if (!ptd.m_idata[IntIdx::withdrawn][i]) {
                                Gpu::Atomic::Add(&infected_family_not_withdrawn_d_ptr[community * max_family + family_ptr[i]], 1);
                                int nc = (community * max_nborhood + nborhood_ptr[i]) * max_nc + family_ptr[i] / FAMILIES_PER_CLUSTER;
                                Gpu::Atomic::Add(&infected_nc_d_ptr[nc], 1);
                            }
                        }
                    });
                    Gpu::synchronize();
                    // Loop to compute infection probability for each susceptible agent.
                    // For each agent, find count of infectious agents in each group and use that as the exponent to compute the
                    // infection probability. In cases where there is an overlap (e.g. infectious agents in same family
                    // and in neighborhood cluster, adjust the infected counts to avoid double-counting the overlap.
                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isHomeCandidate(i, ptd)) {
                            ParticleReal xmit_family_prob = 0;
                            ParticleReal xmit_nc_prob = 0;
                            if (adults) {
                                xmit_family_prob = lparm->xmit_adult[ptd.m_idata[IntIdx::age_group][i]];
                                xmit_nc_prob = lparm->xmit_nc_adult[ptd.m_idata[IntIdx::age_group][i]];
                            } else {
                                xmit_family_prob = lparm->xmit_child[ptd.m_idata[IntIdx::age_group][i]];
                                xmit_nc_prob = lparm->xmit_nc_child[ptd.m_idata[IntIdx::age_group][i]];
                            }
                            auto community = getCommunityIndex(ptd, i);
                            int num_infected_family = infected_family_d_ptr[community * max_family + family_ptr[i]];
                            ParticleReal family_prob = 1.0_prt - infect * xmit_family_prob * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(family_prob, num_infected_family));
                            if (!ptd.m_idata[IntIdx::withdrawn][i]) {
                                int num_infected_family_not_withdrawn =
                                    infected_family_not_withdrawn_d_ptr[community * max_family + family_ptr[i]];
                                AMREX_ALWAYS_ASSERT(num_infected_family >= num_infected_family_not_withdrawn);
                                int nc = (community * max_nborhood + nborhood_ptr[i]) * max_nc + family_ptr[i] / FAMILIES_PER_CLUSTER;
                                int num_infected_nc = infected_nc_d_ptr[nc];
                                AMREX_ALWAYS_ASSERT(num_infected_nc >= num_infected_family_not_withdrawn);
                                ParticleReal nc_prob = 1.0_prt - infect * xmit_nc_prob * scale;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(nc_prob, num_infected_nc - num_infected_family_not_withdrawn));
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
