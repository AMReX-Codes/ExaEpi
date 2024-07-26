/*! @file InteractionModGeneric.H
 * \brief Contains the class describing agent interactions with a simple generic model
 */

#ifndef _INTERACTION_MOD_GENERIC_H_
#define _INTERACTION_MOD_GENERIC_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief Class describing a simple generic agent interactions */
template <typename AC, typename ACT, typename ACTD, typename A>
class InteractionModGeneric : public InteractionModel<AC,ACT,ACTD,A>
{
    public:

        /*! \brief null constructor */
        InteractionModGeneric() { }

        /*! \brief default destructor */
        virtual ~InteractionModGeneric() = default;

        /*! \brief Simulate agent interaction with a simple generic model */
        virtual void interactAgents( AC&, MultiFab& );

        /*! \brief Simulate agent interaction with a simple generic model */
        virtual void interactAgents( AC&, MultiFab&, AC& ) {}

    protected:

    private:
};

/*! \brief Interaction between agents

    Simulate the interactions between agents and compute the infection probability
    for each agent based on the number of infected agents at the same location:

    + Create bins of agents (see #amrex::GetParticleBin, #amrex::DenseBins) with
      their current locations:
      + The bin size is 1 cell
      + #amrex::GetParticleBin maps a particle to its bin index
      + amrex::DenseBins::build() creates the bin-sorted array of particle indices and
        the offset array for each bin (where the offset of a bin is its starting location
        in the bin-sorted array of particle indices).

    + For each bin:
      + Compute the total number of infected agents for each of the two strains.
      + For each agent in the bin, if they are not already infected or immune, infect them
        with a probability of 0.00001 and 0.00002 times the number of infections for each
        strain respectively.
*/
template <typename AC, typename ACT, typename ACTD, typename A>
void InteractionModGeneric<AC,ACT,ACTD,A>::interactAgents( AC& a_agents, /*!< Agent container */
                                                           MultiFab& /* a_mask */ /*!< Masking behavior */)
{
    BL_PROFILE("InteractionModGeneric::interactAgents");
    int n_disease = a_agents.numDiseases();

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            DenseBins<A> bins;
            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            auto& aos   = ptile.GetArrayOfStructs();
            const size_t np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            const Box& box = mfi.validbox();

            int ntiles = numTilesInBox(box, true, bin_size);

            bins.build(np, pstruct_ptr, ntiles, GetParticleBin{plo, dxi, domain, bin_size, box});
            auto inds = bins.permutationPtr();
            auto offsets = bins.offsetsPtr();

            auto& soa   = ptile.GetStructOfArrays();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;

            for (int d = 0; d < n_disease; d++) {

                auto status_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();
                auto strain_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::strain).data();
                auto counter_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();

                amrex::ParallelForRNG( bins.numBins(),
                [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
                {
                    auto cell_start = offsets[i_cell];
                    auto cell_stop  = offsets[i_cell+1];

                    // compute the number of infected in this cell
                    int num_infected[2] = {0, 0};
                    for (unsigned int i = cell_start; i < cell_stop; ++i) {
                        auto pindex = inds[i];
                        if (status_ptr[pindex] == 1) {
                            ++num_infected[strain_ptr[pindex]];
                        }
                    }

                    // second pass - infection prob is propto num_infected
                    for (unsigned int i = cell_start; i < cell_stop; ++i) {
                        auto pindex = inds[i];
                        if ( status_ptr[pindex] != Status::infected &&
                             status_ptr[pindex] != Status::immune) {
                            if (amrex::Random(engine) < 0.0001*num_infected[0]) {
                                strain_ptr[pindex] = 0;
                                status_ptr[pindex] = Status::infected;
                                counter_ptr[pindex] = 0;
                            } else if (amrex::Random(engine) < 0.0002*num_infected[1]) {
                                strain_ptr[pindex] = 1;
                                status_ptr[pindex] = Status::infected;
                                counter_ptr[pindex] = 0;
                            }
                        }
                    }
                });
                amrex::Gpu::synchronize();
            }
        }
    }
}

#endif
