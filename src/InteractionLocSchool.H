/*! @file InteractionLocSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_LOC_SCHOOL_H_
#define _INTERACTION_LOC_SCHOOL_H_

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particles.H>

#include "Locations.H"
#include "AgentDefinitions.H"

using namespace amrex;

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
                                                        MultiFab& a_mask     /*!< Masking behavior */)
{
    BL_PROFILE("InteractionLocSchool::interactAgents");

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::LocationNames::school);

            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            auto& aos   = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            const Box& box = mfi.validbox();
            int ntiles = numTilesInBox(box, true, bin_size);

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, box};
            if (bins_ptr->numBins() < 0) {
                bins_ptr->build(BinPolicy::Serial, np, pstruct_ptr, ntiles, binner);
            }
            AMREX_ALWAYS_ASSERT(np == bins_ptr->numItems());
            Gpu::synchronize();
        }

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

            auto& soa   = ptile.GetStructOfArrays();
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();

            //auto mask_arr = a_mask[mfi].array();

            //auto strain_ptr = soa.GetIntData(IntIdx::strain).data();
            //auto timer_ptr = soa.GetRealData(RealIdx::timer).data();
            auto family_ptr = soa.GetIntData(IntIdx::family).data();
            auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
            auto school_ptr = soa.GetIntData(IntIdx::school).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto prob_ptr = soa.GetRealData(RealIdx::prob).data();
            auto counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();

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

                if (status_ptr[i] == Status::immune) {
                    return;
                }
                if (status_ptr[i] == Status::dead) {
                    return;
                }
                if (status_ptr[i] == Status::infected && counter_ptr[i] < lparm->incubation_length) {
                    // incubation stage
                    return;
                }

                //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                for (unsigned int jj = cell_start; jj < cell_stop; ++jj) {
                    auto j = inds[jj];
                    AMREX_ALWAYS_ASSERT( (Long) j < np);
                    //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
                    if (status_ptr[j] == Status::immune) {
                        continue;
                    }
                    if (status_ptr[j] == Status::dead) {
                        continue;
                    }
                    if (status_ptr[j] == Status::infected && counter_ptr[j] < lparm->incubation_length) {
                        // incubation stage
                        continue;
                    }

                    if ( status_ptr[i] == Status::infected &&
                        (status_ptr[j] != Status::infected && status_ptr[j] != Status::dead)) {

                        // i can infect j
                        Real infect = lparm->infect;
                        infect *= lparm->vac_eff;
                        //infect *= i_mask;
                        //infect *= j_mask;

                        Real social_scale = 1.0;  // TODO this should vary based on cell

                        auto prob = prob_ptr[j];

                        if ( (!withdrawn_ptr[i]) && (!withdrawn_ptr[j]) ) {

                            if (nborhood_ptr[i] == nborhood_ptr[j]) {
                                if (school_ptr[i] == school_ptr[j]) {
                                    if (school_ptr[i] > 5) {
                                        /* Playgroup */
                                        prob *= 1.0 - infect * lparm->xmit_school[6] * social_scale;
                                    } else if (school_ptr[i] == 5) {
                                        /* Day care */
                                        prob *= 1.0 - infect * lparm->xmit_school[5] * social_scale;
                                    }
                                }
                            }
                            /* Elementary/middle/high school in common */
                            if (    (school_ptr[i] == school_ptr[j])
                                 && (school_ptr[i] > 0)
                                 && (school_ptr[i] < 5) ) {
                                if (age_group_ptr[i] <= 1) {  /* Transmitter i is a child */
                                    if (age_group_ptr[j] <= 1) {  /* Receiver j is a child */
                                        prob *= 1.0 - infect * lparm->xmit_school[school_ptr[i]] * social_scale;
                                    } else {  // Child student -> adult teacher/staff transmission
                                        prob  *= 1.0 - infect * lparm->xmit_sch_c2a[school_ptr[i]] * social_scale;
                                    }
                                } else if (age_group_ptr[j] <= 1) {  // Adult teacher/staff -> child student
                                    prob *= 1.0 - infect * lparm->xmit_sch_a2c[school_ptr[i]] * social_scale;
                                }
                            }
                        }

                        Gpu::Atomic::Multiply(&prob_ptr[j], prob);

                    } else if (status_ptr[j] == Status::infected &&
                               (status_ptr[i] != Status::infected && status_ptr[i] != Status::dead)) {

                        if (counter_ptr[j] < lparm->incubation_length) {
                            // TODO: this is already checked above?
                            continue;
                        }

                        // j can infect i
                        Real infect = lparm->infect;
                        infect *= lparm->vac_eff;
                        //infect *= i_mask;
                        //infect *= j_mask;

                        Real social_scale = 1.0;  // TODO this should vary based on cell

                        auto prob = prob_ptr[i];

                        if ( (!withdrawn_ptr[i]) && (!withdrawn_ptr[j]) ) {

                            /* Neighborhood? */
                            if (nborhood_ptr[i] == nborhood_ptr[j]) {
                                if (school_ptr[i] == school_ptr[j]) {
                                    if (school_ptr[i] > 5) {
                                        /* Playgroup */
                                        prob *= 1.0 - infect * lparm->xmit_school[6] * social_scale;
                                    } else if (school_ptr[i] == 5) {
                                        /* Day care */
                                        prob *= 1.0 - infect * lparm->xmit_school[5] * social_scale;
                                    }
                                }
                            }

                            /* Elementary/middle/high school in common */
                            if (    (school_ptr[i] == school_ptr[j])
                                 && (school_ptr[i] > 0)
                                 && (school_ptr[i] < 5) ) {
                                if (age_group_ptr[i] <= 1) {  /* Receiver i is a child */
                                    if (age_group_ptr[j] <= 1) {  /* Transmitter j is a child */
                                        prob *= 1.0 - infect * lparm->xmit_school[school_ptr[i]] * social_scale;
                                    } else {   // Adult teacher/staff -> child student transmission
                                        prob *= 1.0 - infect * lparm->xmit_sch_a2c[school_ptr[i]] * social_scale;
                                    }
                                } else if (age_group_ptr[j] <= 1) {  // Child student -> adult teacher/staff
                                    prob *= 1.0 - infect * lparm->xmit_sch_c2a[school_ptr[i]] * social_scale;
                                }
                            }
                        }

                        Gpu::Atomic::Multiply(&prob_ptr[i], prob);
                    }
                }
            });
            Gpu::synchronize();
        }
    }
}

#endif
