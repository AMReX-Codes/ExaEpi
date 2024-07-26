/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at school. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionSchool ( const int a_i, /*!< Index of infectious agent */
                                      const int a_j, /*!< Index of susceptible agent */
                                      const PTDType& a_ptd, /*!< Particle tile data */
                                      const DiseaseParm* const a_lparm, /*!< disease paramters */
                                      const Real a_social_scale,  /*!< Social scale */
                                      ParticleReal* const a_prob_ptr, /*!< infection probability */
                                      const int a_thread /*!< agent index of this thread */)
{
    Real infect = a_lparm->infect;
    infect *= a_lparm->vac_eff;

    auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
    auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
    auto school_ptr = a_ptd.m_idata[IntIdx::school];
    auto withdrawn_ptr = a_ptd.m_idata[IntIdx::withdrawn];
    if (withdrawn_ptr[a_i] || withdrawn_ptr[a_j]) { return; }
    auto random_travel_ptr = a_ptd.m_idata[IntIdx::random_travel];
    if ((random_travel_ptr[a_i] >= 0) || (random_travel_ptr[a_j] >= 0)) {return;}

    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0_prt;
    if (nborhood_ptr[a_i] == nborhood_ptr[a_j]) {
        if (school_ptr[a_i] == school_ptr[a_j]) {
            if (school_ptr[a_i] > 5) {
                /* Playgroup */
                prob *= 1.0_prt - infect * a_lparm->xmit_school[6] * a_social_scale;
            } else if (school_ptr[a_i] == 5) {
                /* Day care */
                prob *= 1.0_prt - infect * a_lparm->xmit_school[5] * a_social_scale;
            }
        }
    }
    /* Elementary/middle/high school in common */
    if (    (school_ptr[a_i] == school_ptr[a_j])
         && (school_ptr[a_i] > 0)
         && (school_ptr[a_i] < 5) ) {
        if (age_group_ptr[a_i] <= 1) {  /* Transmitter i is a child */
            if (age_group_ptr[a_j] <= 1) {  /* Receiver j is a child */
                prob *= 1.0_prt - infect * a_lparm->xmit_school[school_ptr[a_i]] * a_social_scale;
            } else {  // Child student -> adult teacher/staff transmission
                prob  *= 1.0_prt - infect * a_lparm->xmit_sch_c2a[school_ptr[a_i]] * a_social_scale;
            }
        } else if (age_group_ptr[a_j] <= 1) {  // Adult teacher/staff -> child student
            prob *= 1.0_prt - infect * a_lparm->xmit_sch_a2c[school_ptr[a_i]] * a_social_scale;
        }
    }

    AMREX_ASSERT(a_j == a_thread);
    a_prob_ptr[a_j] *= prob;
}

/*! \brief Class describing agent interactions at school */
template <typename AC, typename ACT, typename ACTD, typename A>
class InteractionModSchool : public InteractionModel<AC,ACT,ACTD,A>
{
    public:

        /*! \brief null constructor */
        InteractionModSchool() { }

        /*! \brief default destructor */
        virtual ~InteractionModSchool() = default;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents( AC&, MultiFab& );

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents( AC&, MultiFab&, AC& ) {};

    protected:

    private:
};

/*! Simulate the interactions between agents at school and compute
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
    + At school - if A and B are in the same school (#IntIdx::school) in the same neighborhood,
      and they are at school:
      + If both A and B are children: use #DiseaseParm::xmit_school
      + If B is a child, and A is an adult, use #DiseaseParm::xmit_sch_c2a
      + If A is a child, and B is an adult, use #DiseaseParm::xmit_sch_a2c
*/
template <typename AC, typename ACT, typename ACTD, typename A>
void InteractionModSchool<AC,ACT,ACTD,A>::interactAgents(AC& a_agents, /*!< Agent container */
                                                         MultiFab& /*a_mask*/ /*!< Masking behavior */)
{
    BL_PROFILE("InteractionModSchool::interactAgents");
    int n_disease = a_agents.numDiseases();

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        this->makeBins( a_agents, bin_size, lev, ExaEpi::InteractionNames::school );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::InteractionNames::school);

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
                    if ( isHospitalized(i, ptd) )  { return; }

                    //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                    for (unsigned int jj = cell_start; jj < cell_stop; ++jj) {

                        auto j = inds[jj];
                        AMREX_ALWAYS_ASSERT( (Long) j < np);
                        //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
                        if (i == j) continue;
                        if ( isHospitalized(j, ptd) )  { continue; }

                        if ( isInfectious<ACTD>(j, ptd, d) ) {
                            Real social_scale = 1.0_prt;  // TODO this should vary based on cell
                            binaryInteractionSchool<ACTD>(  j, i,
                                                            ptd,
                                                            lparm,
                                                            social_scale,
                                                            prob_ptr,
                                                            i );

                        }
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}

#endif
