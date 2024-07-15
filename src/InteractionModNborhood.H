/*! @file InteractionModNborhood.H
 * \brief Contains the class describing agent interactions at neighborhood
 */

#ifndef _INTERACTION_MOD_NBORHOOD_H_
#define _INTERACTION_MOD_NBORHOOD_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent in the neighborhood/community. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionNborhood ( const int a_i, /*!< Index of infectious agent */
                                        const int a_j, /*!< Index of susceptible agent */
                                        const PTDType& a_ptd, /*!< Particle tile data */
                                        const DiseaseParm* const a_lparm, /*!< disease paramters */
                                        const Real a_social_scale, /*!< Social scale */
                                        ParticleReal* const a_prob_ptr, /*!< infection probability */
                                        const bool a_at_work_ptr /*!< whether agents are at work or at home */)
{
    Real infect = a_lparm->infect;
    infect *= a_lparm->vac_eff;

    auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
    auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
    auto school_ptr = a_ptd.m_idata[IntIdx::school];
    auto withdrawn_ptr = a_ptd.m_idata[IntIdx::withdrawn];
    auto work_nborhood_ptr = a_ptd.m_idata[IntIdx::work_nborhood];

    if (withdrawn_ptr[a_i] || withdrawn_ptr[a_j]) { return; }

    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0_prt;

    int nborhood_ai = nborhood_ptr[a_i];
    int nborhood_aj = nborhood_ptr[a_j];

    if (a_at_work_ptr) {
        nborhood_ai = (age_group_ptr[a_i] <= 1) ? nborhood_ptr[a_i] : work_nborhood_ptr[a_i];
        nborhood_aj = (age_group_ptr[a_j] <= 1) ? nborhood_ptr[a_j] : work_nborhood_ptr[a_j];
    }

    // school = 0 means child not going to school (school < 0 not defined anywhere for children)
    // Should always be in the same community = same cell
    if (nborhood_ai == nborhood_aj) {
        if (school_ptr[a_i] < 0 ) { // not attending school, use _SC contacts
            prob *= 1.0_prt - infect * a_lparm->xmit_hood_SC[age_group_ptr[a_j]] * a_social_scale;
        } else {
            prob *= 1.0_prt - infect * a_lparm->xmit_hood[age_group_ptr[a_j]] * a_social_scale;
        }
    } else { // Community?
        if (school_ptr[a_i] < 0) { // not attending school, use _SC contacts
            prob *= 1.0_prt - infect * a_lparm->xmit_comm_SC[age_group_ptr[a_j]] * a_social_scale;
        } else {
            prob *= 1.0_prt - infect * a_lparm->xmit_comm[age_group_ptr[a_j]] * a_social_scale;
        }
    }

    Gpu::Atomic::Multiply(&a_prob_ptr[a_j], prob);
}

/*! \brief Class describing agent interactions in the neighborhood/community */
template <typename AC, typename ACT, typename ACTD, typename A>
class InteractionModNborhood : public InteractionModel<AC,ACT,ACTD,A>
{
    public:

        /*! \brief null constructor */
        InteractionModNborhood() { }

        /*! \brief default destructor */
        virtual ~InteractionModNborhood() = default;

        /*! \brief Simulate agent interaction in the neighborhood/community */
        virtual void interactAgents( AC&, MultiFab& );

    protected:

    private:
};

/*! Simulate the interactions between agents in the neighborhood/community and compute
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
    + Compute probability from community transmission
    + Compute probability from neighborhood
*/
template <typename AC, typename ACT, typename ACTD, typename A>
void InteractionModNborhood<AC,ACT,ACTD,A>::interactAgents( AC& a_agents, /*!< Agent container */
                                                            MultiFab& /*a_mask*/ /*!< Masking behavior */)
{
    BL_PROFILE("InteractionModNborhood::interactAgents");
    int n_disease = a_agents.numDiseases();

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        this->makeBins( a_agents, bin_size, lev, ExaEpi::InteractionNames::nborhood );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::InteractionNames::nborhood);
            auto flag_at_work = a_agents.isAtWork(); /*! Flag to indicate if agents are at work */
            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, mfi.validbox()};
            AMREX_ALWAYS_ASSERT(bins_ptr->numBins() >= 0);
            auto inds = bins_ptr->permutationPtr();
            auto offsets = bins_ptr->offsetsPtr();

            for (int d = 0; d < n_disease; d++) {
                auto prob_ptr = this->getAgentProbPtr(a_agents,lev,mfi,d);
                //auto mask_arr = a_mask[mfi].array();
                auto lparm = a_agents.getDiseaseParameters_d(d);

                ParallelForRNG( bins_ptr->numItems(),
                                [=] AMREX_GPU_DEVICE (int ii, RandomEngine const& /*engine*/)
                                noexcept
                {
                    auto i = inds[ii];
                    int i_cell = binner(pstruct_ptr[i]);
                    auto cell_start = offsets[i_cell];
                    auto cell_stop  = offsets[i_cell+1];

                    AMREX_ALWAYS_ASSERT( (Long) i < np);
                    if ( notSusceptible<ACTD>(i, ptd, d) )  { return; }

                    //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                    for (unsigned int jj = cell_start; jj < cell_stop; ++jj) {

                        auto j = inds[jj];
                        AMREX_ALWAYS_ASSERT( (Long) j < np);

                        //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
                        if (i == j) continue;

                        if ( isInfectious<ACTD>(j, ptd, d) ) {
                            Real social_scale = 1.0_prt;  // TODO this should vary based on cell
                            binaryInteractionNborhood<ACTD>( j, i, ptd, lparm, social_scale, prob_ptr, flag_at_work );
                        }
                    }
                });
                Gpu::synchronize();
            }

        }
    }
}

#endif
