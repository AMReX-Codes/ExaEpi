/*! @file InteractionLocWork.H
 * \brief Contains the class describing agent interactions at work
 */

#ifndef _INTERACTION_LOC_WORK_H_
#define _INTERACTION_LOC_WORK_H_

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particles.H>

#include "Locations.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief Class describing agent interactions at work */
template <typename ACType /*!< agent container type */, typename AType /*!< agent type */>
class InteractionLocWork : public InteractionLocation<ACType,AType>
{
    public:

        /*! \brief null constructor */
        InteractionLocWork() { }

        /*! \brief default destructor */
        virtual ~InteractionLocWork() = default;

        /*! \brief Simulate agent interaction at work */
        virtual void interactAgents( ACType&, MultiFab& );

    protected:

    private:
};

/*! Simulate the interactions between agents at workplace and compute
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
    + At workplace - if agents are at work, and B has a workgroup and work location assigned: If A
      and B have the same workgroup and work location, use the workplace transmission
      probability (#DiseaseParm::xmit_work).
*/
template <typename ACType, typename AType>
void InteractionLocWork<ACType,AType>::interactAgents(ACType&   a_agents,  /*!< Agent container */
                                                      MultiFab& a_mask     /*!< Masking behavior */)
{
    BL_PROFILE("InteractionLocWork::interactAgents");

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        this->makeBins( a_agents, bin_size, lev, ExaEpi::LocationNames::work );

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::LocationNames::work);

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
            auto school_ptr = ( this->m_school_ptr );
            auto work_i_ptr( this->m_work_i_ptr );
            auto workgroup_ptr( this->m_workgroup_ptr );
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

                    if (status_ptr[i] == Status::infected &&
                        (status_ptr[j] != Status::infected && status_ptr[j] != Status::dead)) {

                        // i can infect j
                        Real infect = lparm->infect;
                        infect *= lparm->vac_eff;
                        //infect *= i_mask;
                        //infect *= j_mask;

                        Real social_scale = 1.0;  // TODO this should vary based on cell
                        Real work_scale = 1.0;  // TODO this should vary based on cell

                        auto prob = prob_ptr[j];
                        // /* Home isolation or household quarantine? */
                        if ( (!withdrawn_ptr[i]) && (!withdrawn_ptr[j]) ) {
                            // school < 0 means a child normally attends school, but not today
                            /* Should always be in the same community = same cell */
                            if (school_ptr[i] < 0) {  // not attending school, use _SC contacts
                                prob *= 1.0 - infect * lparm->xmit_comm_SC[age_group_ptr[j]] * social_scale;
                            } else {
                                prob *= 1.0 - infect * lparm->xmit_comm[age_group_ptr[j]] * social_scale;
                            }
                            /* Workgroup transmission */
                            if (workgroup_ptr[i] && (work_i_ptr[i] >= 0)) { // transmitter i at work
                                if ((work_i_ptr[j] >= 0) && (workgroup_ptr[i] == workgroup_ptr[j])) {  // coworker
                                    prob *= 1.0 - infect * lparm->xmit_work * work_scale;
                                }
                            }
                            // /* Neighborhood? */
                            if (nborhood_ptr[i] == nborhood_ptr[j]) {
                                if (school_ptr[i] < 0)  {
                                    // not attending school, use _SC contacts
                                    prob *= 1.0 - infect * lparm->xmit_hood_SC[age_group_ptr[j]] * social_scale;
                                } else {
                                    prob *= 1.0 - infect * lparm->xmit_hood[age_group_ptr[j]] * social_scale;
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
                        Real work_scale = 1.0;  // TODO this should vary based on cell

                        auto prob = prob_ptr[i];
                        /* Home isolation or household quarantine? */  // TODO - be careful about withdrawn versus at home...
                        if ( (!withdrawn_ptr[i]) && (!withdrawn_ptr[j]) ) {
                            // school < 0 means a child normally attends school, but not today
                            /* Should always be in the same community = same cell */
                            if (school_ptr[j] < 0) {  // not attending school, use _SC contacts
                                prob *= 1.0 - infect * lparm->xmit_comm_SC[age_group_ptr[i]] * social_scale;
                            } else {
                                prob *= 1.0 - infect * lparm->xmit_comm[age_group_ptr[i]] * social_scale;
                            }
                            /* Workgroup transmission */
                            if (workgroup_ptr[j] && (work_i_ptr[j] >= 0)) { // transmitter j at work
                                if ((work_i_ptr[i] >= 0) && (workgroup_ptr[i] == workgroup_ptr[j])) {  // coworker
                                    prob *= 1.0 - infect * lparm->xmit_work * work_scale;
                                }
                            }
                            /* Neighborhood? */
                            if (nborhood_ptr[i] == nborhood_ptr[j]) {
                                if (school_ptr[j] < 0) { // not attending school, use _SC contacts
                                    prob *= 1.0 - infect * lparm->xmit_hood_SC[age_group_ptr[i]] * social_scale;
                                } else {
                                    prob *= 1.0 - infect * lparm->xmit_hood[age_group_ptr[i]] * social_scale;
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
