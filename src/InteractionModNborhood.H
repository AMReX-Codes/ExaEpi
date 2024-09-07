/*! @file InteractionModNborhood.H
 * \brief Contains the class describing agent interactions at neighborhood
 */

#ifndef _INTERACTION_MOD_NBORHOOD_H_
#define _INTERACTION_MOD_NBORHOOD_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


template <typename PTDType>
struct NborhoodBinner
{
    NborhoodBinner(const Geometry &geom, const IntVect &bin_size, const Box &box) :
               bin_size(bin_size), box(box) {
        dxi = geom.InvCellSizeArray();
        plo = geom.ProbLoArray();
        domain = geom.Domain();
    }

    AMREX_GPU_HOST_DEVICE
    unsigned int operator() (const PTDType& ptd, int i) const noexcept
    {
        Box tbx;
        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
        auto tid = getTileIndex(iv, box, true, bin_size, tbx);
        return static_cast<unsigned int>(tid);
    }

  private:

    GpuArray<Real,AMREX_SPACEDIM> plo;
    GpuArray<Real,AMREX_SPACEDIM> dxi;
    Box domain;
    IntVect bin_size;
    Box box;
};


/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent in the neighborhood/community. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionNborhood (const int infectious_i, /*!< Index of infectious agent */
                                        const int susceptible_i, /*!< Index of susceptible agent */
                                        const PTDType& a_ptd, /*!< Particle tile data */
                                        const DiseaseParm* const a_lparm, /*!< disease paramters */
                                        const Real a_social_scale, /*!< Social scale */
                                        ParticleReal* const a_prob_ptr, /*!< infection probability */
                                        const bool a_at_work_ptr /*!< whether agents are at work or at home */)
{
    Real infect = a_lparm->infect;
    infect *= (1.0_rt - a_lparm->vac_eff);

    auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
    auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
    auto work_nborhood_ptr = a_ptd.m_idata[IntIdx::work_nborhood];
    auto school_ptr = a_ptd.m_idata[IntIdx::school];

    if (a_at_work_ptr) {
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);
    } else {
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::home_i][infectious_i] == a_ptd.m_idata[IntIdx::home_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::home_j][infectious_i] == a_ptd.m_idata[IntIdx::home_j][susceptible_i]);
    }

    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0_prt;
    // define neighbordhood based on whether agent is in home or work community
    int nborhood_infectious_i = nborhood_ptr[infectious_i];
    int nborhood_susceptible_i = nborhood_ptr[susceptible_i];
    if (a_at_work_ptr) {
        if (age_group_ptr[infectious_i] > 1) nborhood_infectious_i = work_nborhood_ptr[infectious_i];
        if (age_group_ptr[susceptible_i] > 1) nborhood_susceptible_i = work_nborhood_ptr[susceptible_i];
    }

    // school < 0 means a child normally attends school, but not today
    // Should always be in the same community = same cell
    // Neighborhood?
    if (nborhood_infectious_i == nborhood_susceptible_i) {
        if (school_ptr[infectious_i] < 0 ) { // not attending school, use _SC contacts
            prob *= 1.0_prt - infect * a_lparm->xmit_hood_SC[age_group_ptr[susceptible_i]] * a_social_scale;
        } else {
            prob *= 1.0_prt - infect * a_lparm->xmit_hood[age_group_ptr[susceptible_i]] * a_social_scale;
        }
    } else { // Community?
        if (school_ptr[infectious_i] < 0) { // not attending school, use _SC contacts
            prob *= 1.0_prt - infect * a_lparm->xmit_comm_SC[age_group_ptr[susceptible_i]] * a_social_scale;
        } else {
            prob *= 1.0_prt - infect * a_lparm->xmit_comm[age_group_ptr[susceptible_i]] * a_social_scale;
        }
    }
    Gpu::Atomic::Multiply(&a_prob_ptr[susceptible_i], prob);
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
        virtual void interactAgents( AC&, MultiFab& ) override;

        /*! \brief Simulate agent interaction in the neighborhood/community */
        virtual void interactAgents( AC&, MultiFab&, AC& ) override {
            amrex::Abort("Do not use this interface for this interaction model");
        }

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
void InteractionModNborhood<AC,ACT,ACTD,A>::interactAgents (AC& a_agents, /*!< Agent container */
                                                            MultiFab& /*a_mask*/ /*!< Masking behavior */)
{
    BL_PROFILE("InteractionModNborhood::interactAgents");
    int n_disease = a_agents.numDiseases();
    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};

    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto num_tiles = numTilesInBox(mfi.validbox(), true, bin_size);
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();

            auto flag_at_work = a_agents.isAtWork(); /*! Flag to indicate if agents are at work */

            NborhoodBinner<ACTD> binner(a_agents.Geom(lev), bin_size, mfi.validbox());
            DenseBins<ACTD> bins_nborhood;
            bins_nborhood.build(BinPolicy::GPU, np, ptd, num_tiles, binner);

            AMREX_ALWAYS_ASSERT(bins_nborhood.numBins() >= 0);
            auto inds = bins_nborhood.permutationPtr();
            auto offsets = bins_nborhood.offsetsPtr();

            for (int d = 0; d < n_disease; d++) {
                auto prob_ptr = this->getAgentProbPtr(a_agents, lev, mfi, d);
                //auto mask_arr = a_mask[mfi].array();
                auto lparm = a_agents.getDiseaseParameters_d(d);
                Real social_scale = 1.0_prt;  // TODO this should vary based on cell

                ParallelFor(bins_nborhood.numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept {
                    auto infectious_i = inds[ii];
                    AMREX_ALWAYS_ASSERT((Long)infectious_i < np);
                    int i_cell = binner(ptd, infectious_i);
                    auto cell_start = offsets[i_cell];
                    auto cell_stop  = offsets[i_cell + 1];

                    if (isInfectious(infectious_i, ptd, d) &&  // infectious
                        !isHospitalized(infectious_i, ptd) &&  // not in hospital
                        !withdrawn_ptr[infectious_i] &&        // not withdrawn
                        random_travel_ptr[infectious_i] < 0)   // not traveling
                    {
                        //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                        for (auto jj = cell_start; jj < cell_stop; ++jj) {
                            auto susceptible_i = inds[jj];
                            AMREX_ALWAYS_ASSERT((Long)susceptible_i < np);
                            //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
                            if (infectious_i != susceptible_i &&
                                isSusceptible(susceptible_i, ptd, d) &&
                                !isHospitalized(susceptible_i, ptd) &&
                                !withdrawn_ptr[susceptible_i] &&
                                random_travel_ptr[susceptible_i] < 0)
                            {
                                binaryInteractionNborhood<ACTD>(infectious_i, susceptible_i, ptd, lparm, social_scale, prob_ptr,
                                                                flag_at_work);
                            }
                        }
                    }
                });
                Gpu::synchronize();
            }

        }
    }
}

#endif
