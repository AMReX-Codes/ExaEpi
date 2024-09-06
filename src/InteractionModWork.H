/*! @file InteractionModWork.H
 * \brief Contains the class describing agent interactions at work
 */

#ifndef _INTERACTION_MOD_WORK_H_
#define _INTERACTION_MOD_WORK_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


template <typename PTDType>
struct WorkBinner
{
    WorkBinner(const Geometry &geom, const IntVect &bin_size, const Box &box, const int max_wg) :
            bin_size(bin_size), box(box), max_wg(max_wg) {
        dxi = geom.InvCellSizeArray();
        plo = geom.ProbLoArray();
        domain = geom.Domain();
    }

    AMREX_GPU_HOST_DEVICE
    unsigned int operator() (const PTDType& ptd, int i) const noexcept {
        Box tbx;
        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
        auto tid = getTileIndex(iv, box, true, bin_size, tbx);
        auto wg = ptd.m_idata[IntIdx::workgroup][i];
        return static_cast<unsigned int>(tid * max_wg + wg);
    }

  private:

    GpuArray<Real,AMREX_SPACEDIM> plo;
    GpuArray<Real,AMREX_SPACEDIM> dxi;
    Box domain;
    IntVect bin_size;
    Box box;
    int max_wg;
};

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at work. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionWork (const int infectious_i, /*!< Index of infectious agent */
                                   const int susceptible_i, /*!< Index of susceptible agent */
                                   const PTDType& a_ptd, /*!< Particle tile data */
                                   const DiseaseParm* const a_lparm,  /*!< disease paramters */
                                   const Real a_work_scale, /*!< work scale */
                                   ParticleReal* const a_prob_ptr, /*!< infection probability */
                                   const int a_d   /*!< disease index */)
{
    Real infect = a_lparm->infect;
    infect *= (1.0_rt - a_lparm->vac_eff);

    auto withdrawn_ptr = a_ptd.m_idata[IntIdx::withdrawn];
    auto random_travel_ptr = a_ptd.m_idata[IntIdx::random_travel];

    //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
    if (infectious_i == susceptible_i) return;          // same agents
    if (!isSusceptible(susceptible_i, a_ptd, a_d)) return; // not susceptible
    if (isHospitalized(susceptible_i, a_ptd)) return;      // in hospital
    if (!isAtWork(susceptible_i, a_ptd)) return;        // not at work
    if (withdrawn_ptr[susceptible_i]) return;           // withdrawn
    if (random_travel_ptr[susceptible_i] >= 0) return;  // on random travel

    AMREX_ASSERT(a_ptd.m_idata[IntIdx::workgroup][infectious_i] == a_ptd.m_idata[IntIdx::workgroup][susceptible_i]);
    AMREX_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                 a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);
    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0_prt - infect * a_lparm->xmit_work * a_work_scale;
    // this is needed if we find all the susceptible for each infectious in turn
    Gpu::Atomic::Multiply(&a_prob_ptr[susceptible_i], prob);
}

/*! \brief Class describing agent interactions at work */
template <typename AC, typename ACT, typename ACTD, typename A>
class InteractionModWork : public InteractionModel<AC,ACT,ACTD,A>
{
    public:

        /*! \brief null constructor */
        InteractionModWork() { }

        /*! \brief default destructor */
        virtual ~InteractionModWork() = default;

        /*! \brief Simulate agent interaction at work */
        virtual void interactAgents( AC&, MultiFab& ) override;

        /*! \brief Simulate agent interaction at work */
        virtual void interactAgents( AC&, MultiFab&, AC& ) override {
            amrex::Abort("Do not use this interface for this interaction model");
        }

    protected:

    private:

        DenseBins<ACTD> bins_work;
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
    + At workplace - if agents are at work, and B has a workgroup and work location assigned: If A
      and B have the same workgroup and work location, use the workplace transmission
      probability (#DiseaseParm::xmit_work).
*/
template <typename AC, typename ACT, typename ACTD, typename A>
void InteractionModWork<AC,ACT,ACTD,A>::interactAgents(AC& a_agents,
                                                       MultiFab&)
{
    BL_PROFILE("InteractionModWork::interactAgents");
    int n_disease = a_agents.numDiseases();

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto num_tiles = numTilesInBox(mfi.validbox(), true, bin_size);

            int max_wg = 0;
            for (int i = 0; i < np; i++) {
                max_wg = max(max_wg, workgroup_ptr[i]);
            }

            // agents are binned by community + workgroup. These values can change from one time step to the next so they
            // have to be recomputed (rebuild the bins) every time step
            WorkBinner<ACTD> binner(a_agents.Geom(lev), bin_size, mfi.validbox(), max_wg);
            bins_work.build(BinPolicy::Serial, np, ptd, num_tiles * max_wg, binner);

            AMREX_ALWAYS_ASSERT(bins_work.numBins() >= 0);
            auto inds = bins_work.permutationPtr();
            auto offsets = bins_work.offsetsPtr();

            for (int d = 0; d < n_disease; d++) {
                auto prob_ptr = this->getAgentProbPtr(a_agents, lev, mfi, d);
                //auto mask_arr = a_mask[mfi].array();
                auto lparm = a_agents.getDiseaseParameters_d(d);
                Real work_scale = 1.0_prt;  // TODO this should vary based on cell
                Real infect = lparm->infect;
                infect *= (1.0_rt - lparm->vac_eff);

                ParallelFor(bins_work.numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept {
                    auto infectious_i = inds[ii];
                    int i_cell = binner(ptd, infectious_i);
                    auto cell_start = offsets[i_cell];
                    auto cell_stop = offsets[i_cell + 1];
                    AMREX_ALWAYS_ASSERT((Long)infectious_i < np);
                    // Doing infectious first requires an atomic operation for GPUs, but generally requires far fewer operations
                    // because there are usually more susceptible agents than infectious. This can be a large performance
                    // difference for CPU only runs.
                    if (isInfectious(infectious_i, ptd, d) &&  // infectious
                        !isHospitalized(infectious_i, ptd) &&  // not in hospital
                        isAtWork(infectious_i, ptd) &&         // worker at work
                        !withdrawn_ptr[infectious_i] &&        // not withdrawn
                        random_travel_ptr[infectious_i] < 0)   // not traveling
                    {
                        //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                        for (auto jj = cell_start; jj < cell_stop; ++jj) {
                            auto susceptible_i = inds[jj];
                            AMREX_ALWAYS_ASSERT((Long)susceptible_i < np);
                            binaryInteractionWork(infectious_i, susceptible_i, ptd, lparm, work_scale, prob_ptr, d);
                        }
                    }
                });
                Gpu::synchronize();
            }
        }
    }

}

#endif
