/*! @file InteractionLocSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_LOC_SCHOOL_H_
#define _INTERACTION_LOC_SCHOOL_H_

#include "Locations.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at school. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionSchool ( const int a_i,  /*!< Index of infectious agent */
                                      const int a_j,  /*!< Index of susceptible agent */
                                      const DiseaseParm* const a_lparm,   /*!< disease paramters */
                                      const Real a_social_scale,  /*!< Social scale */
                                      const int* const a_age_group_ptr,   /*!< age group */
                                      const int* const a_nborhood_ptr,    /*!< neighborhood */
                                      const int* const a_school_ptr,      /*!< school */
                                      ParticleReal* const a_prob_ptr      /*!< infection probability */)
{
    Real infect = a_lparm->infect;
    infect *= a_lparm->vac_eff;
    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0;
    if (a_nborhood_ptr[a_i] == a_nborhood_ptr[a_j]) {
        if (a_school_ptr[a_i] == a_school_ptr[a_j]) {
            if (a_school_ptr[a_i] > 5) {
                /* Playgroup */
                prob *= 1.0 - infect * a_lparm->xmit_school[6] * a_social_scale;
            } else if (a_school_ptr[a_i] == 5) {
                /* Day care */
                prob *= 1.0 - infect * a_lparm->xmit_school[5] * a_social_scale;
            }
        }
    }
    /* Elementary/middle/high school in common */
    if (    (a_school_ptr[a_i] == a_school_ptr[a_j])
         && (a_school_ptr[a_i] > 0)
         && (a_school_ptr[a_i] < 5) ) {
        if (a_age_group_ptr[a_i] <= 1) {  /* Transmitter i is a child */
            if (a_age_group_ptr[a_j] <= 1) {  /* Receiver j is a child */
                prob *= 1.0 - infect * a_lparm->xmit_school[a_school_ptr[a_i]] * a_social_scale;
            } else {  // Child student -> adult teacher/staff transmission
                prob  *= 1.0 - infect * a_lparm->xmit_sch_c2a[a_school_ptr[a_i]] * a_social_scale;
            }
        } else if (a_age_group_ptr[a_j] <= 1) {  // Adult teacher/staff -> child student
            prob *= 1.0 - infect * a_lparm->xmit_sch_a2c[a_school_ptr[a_i]] * a_social_scale;
        }
    }
    Gpu::Atomic::Multiply(&a_prob_ptr[a_j], prob);
}

/*! \brief Class describing agent interactions at school */
template <typename ACType /*!< agent container type */, typename AType /*!< agent type */>
class InteractionLocSchool : public InteractionLocation<ACType,AType>
{
    public:

        /*! \brief null constructor */
        InteractionLocSchool() { }

        /*! \brief default destructor */
        virtual ~InteractionLocSchool() = default;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents( ACType&, MultiFab& );

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
      + If the agent is #Status::infected with the number of days infected (RealIdx::disease_counter)
        less than the #DiseaseParm::incubation_length, do nothing.
      + Else, for each agent *j* in the same bin:
        + If the agent is #Status::immune, do nothing.
        + If the agent is #Status::infected with the number of days infected (RealIdx::disease_counter)
          less than the #DiseaseParm::incubation_length, do nothing.
        + If *i* is infected and *j* is not infected, compute probability of *j* getting infected from i
          (see below).
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
template <typename ACType, typename AType>
void InteractionLocSchool<ACType,AType>::interactAgents(ACType&   a_agents,  /*!< Agent container */
                                                        MultiFab& /*a_mask*/     /*!< Masking behavior */)
{
    BL_PROFILE("InteractionLocSchool::interactAgents");

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        this->makeBins( a_agents, bin_size, lev, ExaEpi::LocationNames::school );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::LocationNames::school);

            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            auto& aos   = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, mfi.validbox()};
            AMREX_ALWAYS_ASSERT(bins_ptr->numBins() >= 0);
            auto inds = bins_ptr->permutationPtr();
            auto offsets = bins_ptr->offsetsPtr();

            this->getAgentAttribs( a_agents, lev, mfi );
            auto status_ptr( this->m_status_ptr );
            auto age_group_ptr( this->m_age_group_ptr );
            auto nborhood_ptr( this->m_nborhood_ptr );
            auto school_ptr( this->m_school_ptr );
            auto withdrawn_ptr( this->m_withdrawn_ptr );
            auto counter_ptr( this->m_counter_ptr );
            //auto strain_ptr( this->m_strain_ptr );
            //auto timer_ptr( this->m_timer_ptr );

            auto prob_ptr( this->m_prob_ptr );

            //auto mask_arr = a_mask[mfi].array();
            auto lparm = a_agents.getDiseaseParm_d();

            ParallelForRNG( bins_ptr->numItems(),
                            [=] AMREX_GPU_DEVICE (int ii, RandomEngine const& /*engine*/)
                            noexcept
            {
                auto i = inds[ii];
                int i_cell = binner(pstruct_ptr[i]);
                auto cell_start = offsets[i_cell];
                auto cell_stop  = offsets[i_cell+1];

                AMREX_ALWAYS_ASSERT( (Long) i < np);

                if ( notInfectiousButInfected( i, status_ptr, counter_ptr, lparm->incubation_length )
                     || notSusceptible( i, status_ptr ) )  {
                    return;
                }
                if (withdrawn_ptr[i]) {
                    return;
                }

                //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                for (unsigned int jj = cell_start; jj < cell_stop; ++jj) {
                    auto j = inds[jj];
                    AMREX_ALWAYS_ASSERT( (Long) j < np);
                    //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);

                    if ( notInfectiousButInfected( j, status_ptr,counter_ptr,lparm->incubation_length )
                         || notSusceptible( j, status_ptr ) )  {
                        continue;
                    }
                    if (withdrawn_ptr[j]) {
                        continue;
                    }

                    if ( isInfectious( i, status_ptr,counter_ptr,lparm->incubation_length )
                         && isSusceptible( j, status_ptr ) ) {

                        Real social_scale = 1.0;  // TODO this should vary based on cell
                        binaryInteractionSchool(  i, j,
                                                  lparm, social_scale,
                                                  age_group_ptr,
                                                  nborhood_ptr,
                                                  school_ptr,
                                                  prob_ptr );

                    } else if ( isInfectious( j, status_ptr,counter_ptr,lparm->incubation_length )
                                && isSusceptible( i, status_ptr ) ) {

                        Real social_scale = 1.0;  // TODO this should vary based on cell
                        binaryInteractionSchool(  j, i,
                                                  lparm, social_scale,
                                                  age_group_ptr,
                                                  nborhood_ptr,
                                                  school_ptr,
                                                  prob_ptr );

                    }
                }
            });
            Gpu::synchronize();
        }
    }
}

#endif
