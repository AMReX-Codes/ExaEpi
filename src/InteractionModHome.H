/*! @file InteractionModHome.H
 * \brief Contains the class describing agent interactions at home
 */

#ifndef _INTERACTION_MOD_HOME_H_
#define _INTERACTION_MOD_HOME_H_

#include "InteractionModel.H"
#include "DiseaseParm.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at home. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionHome (const int infectious_i, /*!< Index of infectious agent */
                                   const int susceptible_i, /*!< Index of susceptible agent */
                                   const PTDType& a_ptd, /*!< Particle tile data */
                                   const DiseaseParm* const a_lparm,  /*!< disease paramters */
                                   const Real a_social_scale, /*!< Social scale */
                                   ParticleReal* const a_prob_ptr, /*!< infection probability */
                                   const int a_d   /*!< disease index */)
{
    Real infect = a_lparm->infect;
    infect *= (1.0_rt - a_lparm->vac_eff);

    auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
    auto family_ptr = a_ptd.m_idata[IntIdx::family];
    auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
    auto school_ptr = a_ptd.m_idata[IntIdx::school];
    auto withdrawn_ptr = a_ptd.m_idata[IntIdx::withdrawn];
    auto random_travel_ptr = a_ptd.m_idata[IntIdx::random_travel];

    if (infectious_i == susceptible_i) return;          // same agents
    if (!isSusceptible(susceptible_i, a_ptd, a_d)) return; // not susceptible
    if (isHospitalized(susceptible_i, a_ptd)) return;      // in hospital
    if (random_travel_ptr[susceptible_i] >= 0) return;  // on random travel
    if (nborhood_ptr[infectious_i] != nborhood_ptr[susceptible_i]) return;  // different neighborhood
    if (family_ptr[infectious_i] / 4 != family_ptr[susceptible_i] / 4) return; // different family cluster

    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0_prt;
    if (family_ptr[infectious_i] == family_ptr[susceptible_i]) {
        // at home, within a family
        if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child */
            if (school_ptr[infectious_i] < 0) { // not attending school, use _SC contacts
                prob *= 1.0_prt - infect * a_lparm->xmit_child_SC[age_group_ptr[susceptible_i]];
            } else {
                prob *= 1.0_prt - infect * a_lparm->xmit_child[age_group_ptr[susceptible_i]];
            }
        } else {
            if (school_ptr[infectious_i] < 0) { // not attending school, use _SC contacts
                prob *= 1.0_prt - infect * a_lparm->xmit_adult_SC[age_group_ptr[susceptible_i]];
            } else {
                prob *= 1.0_prt - infect * a_lparm->xmit_adult[age_group_ptr[susceptible_i]];
            }
        }
    } else if (!withdrawn_ptr[infectious_i] && !withdrawn_ptr[susceptible_i]) {
        // not withdrawn, within a family cluster
        if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
            if (school_ptr[infectious_i] < 0)  // not attending school, use _SC contacts
                prob *= 1.0_prt - infect * a_lparm->xmit_nc_child_SC[age_group_ptr[susceptible_i]] * a_social_scale;
            else
                prob *= 1.0_prt - infect * a_lparm->xmit_nc_child[age_group_ptr[susceptible_i]] * a_social_scale;
        } else {
            if (school_ptr[infectious_i] < 0)  {   // not attending school, use _SC contacts
                prob *= 1.0_prt - infect * a_lparm->xmit_nc_adult_SC[age_group_ptr[susceptible_i]] * a_social_scale;
            } else {
                prob *= 1.0_prt - infect * a_lparm->xmit_nc_adult[age_group_ptr[susceptible_i]] * a_social_scale;
            }
        }
    }
    // atomic is needed if we find all the susceptible for each infectious in turn
    Gpu::Atomic::Multiply(&a_prob_ptr[susceptible_i], prob);
}

/*! \brief Class describing agent interactions at home */
template <typename AC, typename ACT, typename ACTD, typename A>
class InteractionModHome : public InteractionModel<AC,ACT,ACTD,A>
{
    public:

        /*! \brief null constructor */
        InteractionModHome() { }

        /*! \brief default destructor */
        virtual ~InteractionModHome() = default;

        /*! \brief Simulate agent interaction at home */
        virtual void interactAgents( AC&, MultiFab& ) override;

        /*! \brief Simulate agent interaction at home */
        virtual void interactAgents( AC&, MultiFab&, AC& ) override {
            amrex::Abort("Do not use this interface for this interaction model");
        }

    protected:

    private:
};

/*! Simulate the interactions between agents at home and compute
    the infection probability for each agent:

    + Create bins of agents if not already created (see
      #amrex::GetParticleBin, #amrex::DenseBins):
      + The bin size is 1 cell
      + #amrex::GetParticleBin maps a particle to its bin index
      + amrex::DenseBins::build() creates the bin-sorted array of particle indices and
        the offset array for each bin (where the offset of a bin is its starting location
        in the bin-sorted array of particle indices).

    + For each agent *i* in the bin-sorted array of agents:
      + Find its bin and the range of indices in the bin-sorted array for agents in its bin
      + If the agent is #Status::immune, do nothing.
      + If the agent is #Status::infected with the number of days infected (RealIdxDisease::disease_counter)
        less than the incubation length, do nothing.
      + Else, for each agent *j* in the same bin:
        + If the agent is #Status::immune, do nothing.
        + If the agent is #Status::infected with the number of days infected (RealIdxDisease::disease_counter)
          less than the incubation length, do nothing.
        + Else if *i* is not infected and *j* is infected, compute probability of *i* getting infected
          from *j* (see below).

    Summary of how the probability of agent A getting infected from agent B is computed:
    + Compute infection probability reduction factor from vaccine efficacy (#DiseaseParm::vac_eff)
    + Within family - if their IntIdx::nborhood and IntIdx::family indices are same,
      and the agents are at home:
      + If B is a child, use the appropriate transmission probability (#DiseaseParm::xmit_child_SC or
        #DiseaseParm::xmit_child) depending on whether B goes to school or not (#IntIdx::school)
      + If B is an adult, use the appropriate transmission probability (#DiseaseParm::xmit_adult_SC or
        #DiseaseParm::xmit_adult) depending on whether B works at a school or not (#IntIdx::school)
*/
template <typename AC, typename ACT, typename ACTD, typename A>
void InteractionModHome<AC,ACT,ACTD,A>::interactAgents(AC& a_agents, /*!< Agent container */
                                                       MultiFab& /*a_mask*/ /*!< Masking behavior */)
{
    BL_PROFILE("InteractionModHome::interactAgents");
    int n_disease = a_agents.numDiseases();

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();
        //this->makeBins(a_agents, bin_size, lev, ExaEpi::InteractionNames::home );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto num_tiles = numTilesInBox(mfi.validbox(), true, bin_size);
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::InteractionNames::home);
            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();
            auto& soa = ptile.GetStructOfArrays();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, mfi.validbox()};

            //auto binner = GetWorkerBin<A>{plo, dxi, domain, bin_size, mfi.validbox(), max_wg};
            bins_ptr->build(BinPolicy::Serial, np, pstruct_ptr, num_tiles, binner);

            AMREX_ALWAYS_ASSERT(bins_ptr->numBins() >= 0);
            auto inds = bins_ptr->permutationPtr();
            auto offsets = bins_ptr->offsetsPtr();

            for (int d = 0; d < n_disease; d++) {

                auto prob_ptr = this->getAgentProbPtr(a_agents,lev,mfi,d);
                //auto mask_arr = a_mask[mfi].array();
                auto lparm = a_agents.getDiseaseParameters_d(d);

                ParallelFor(bins_ptr->numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept {
                    auto infectious_i = inds[ii];
                    int i_cell = binner(pstruct_ptr[infectious_i]);
                    auto cell_start = offsets[i_cell];
                    auto cell_stop  = offsets[i_cell+1];

                    AMREX_ALWAYS_ASSERT((Long)infectious_i < np);

                    if (isInfectious(infectious_i, ptd, d) &&  // infectious
                        !isHospitalized(infectious_i, ptd) &&  // not in hospital
                        random_travel_ptr[infectious_i] < 0)   // not traveling
                    {
                        //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                        for (auto jj = cell_start; jj < cell_stop; ++jj) {
                            auto susceptible_i = inds[jj];
                            AMREX_ALWAYS_ASSERT((Long)susceptible_i < np);
                            //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
                            Real social_scale = 1.0_prt;  // TODO this should vary based on cell
                            binaryInteractionHome<ACTD>(infectious_i, susceptible_i, ptd, lparm, social_scale, prob_ptr, d);
                        }
                    }
                });
                Gpu::synchronize();
            }

        }
    }
}

#endif
