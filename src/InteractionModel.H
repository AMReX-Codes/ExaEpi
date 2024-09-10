/*! @file InteractionModel.H
    \brief Defines interaction model base class */

#ifndef _INTERACTION_MODEL_H_
#define _INTERACTION_MODEL_H_

#include <map>
#include <string>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include "AgentDefinitions.H"

using namespace amrex;

namespace ExaEpi
{
    /*! \brief Name of models */
    namespace InteractionNames
    {
        const std::string generic = "generic";
        const std::string home = "home";
        const std::string work = "work";
        const std::string school = "school";
        const std::string nborhood = "neighborhood";
        const std::string transit = "transit";
        const std::string random = "random";
    }
}

/*! \brief Base class for defining interaction models
 *
 *  Contains things that are common to all interaction model classes.
 *  + The member data pointers are used to temporarily point to the SoA attributes of
 *    the agents for a given level and MFIter; for the various interaction models
 *    inheriting from this class, InteractionModel::getAgentAttribs() is called
 *    to set these pointers before using them.
 */
template <  typename AC /*!< agent container type */,
            typename ACT, /*!< agent container tile type */
            typename ACTD, /*!< agent container tile data type */
            typename A /*!< agent type */>
class InteractionModel
{
    public:

        /*! \brief null constructor */
        InteractionModel<AC,ACT,ACTD,A>() { }

        /*! \brief default destructor */
        virtual ~InteractionModel<AC,ACT,ACTD,A>() = default;

        /*! \brief Interact agents for a model */
        virtual void interactAgents(AC&, MultiFab&) = 0;

        /*! \brief Interact agents for a model */
        virtual void interactAgents(AC&, MultiFab&, AC&) = 0;

    protected:


        /*! \brief Get probability pointer from agent container */
        inline ParticleReal* getAgentProbPtr (  AC&           a_agents,  /*!< agent container */
                                                const int     a_lev,     /*!< level */
                                                const MFIter& a_mfi,     /*!< multifab iterator*/
                                                const int     a_d_idx    /*!< disease index */ )
        {
            BL_PROFILE("InteractionModel::getAgentProbPtr");
            auto& ptile = a_agents.ParticlesAt(a_lev, a_mfi);
            auto& soa   = ptile.GetStructOfArrays();
            int r_RT = RealIdx::nattribs;
            return soa.GetRealData(r_RT+r0(a_d_idx)+RealIdxDisease::prob).data();
        }

        /*! \brief Make particle bins for interactions at a given model */
        virtual void makeBins(  AC&                 a_agents,   /*!< agent container */
                                const IntVect&      a_bin_size, /*!< bin size */
                                const int           a_lev,      /*!< level */
                                const std::string&  a_mod       /*!< interaction model */ )
        {
            BL_PROFILE("InteractionModel::makeBins");
            const Geometry& geom = a_agents.Geom(a_lev);
            const auto dxi = geom.InvCellSizeArray();
            const auto plo = geom.ProbLoArray();
            const auto domain = geom.Domain();

            for(MFIter mfi = a_agents.MakeMFIter(a_lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
                auto bins_ptr = a_agents.getBins(pair_ind, a_mod);

                auto& ptile = a_agents.ParticlesAt(a_lev, mfi);
                auto& aos   = ptile.GetArrayOfStructs();
                const auto np = aos.numParticles();
                const auto& ptd = ptile.getParticleTileData();

                const Box& box = mfi.validbox();
                int ntiles = numTilesInBox(box, true, a_bin_size);

                auto binner = GetParticleBin{plo, dxi, domain, a_bin_size, box};
                bins_ptr->build(BinPolicy::Serial, np, ptd, ntiles, binner);

                AMREX_ALWAYS_ASSERT(np == bins_ptr->numItems());

                Gpu::synchronize();
            }
        }

    private:
};

template <typename PTDType>
struct Binner
{
    Binner(const Geometry &geom, const IntVect &bin_size, const Box &box, const int max_group, const int bin_idx) :
           bin_size(bin_size), box(box), max_group(max_group), bin_idx(bin_idx) {
        dxi = geom.InvCellSizeArray();
        plo = geom.ProbLoArray();
        domain = geom.Domain();
    }

    AMREX_GPU_HOST_DEVICE
    unsigned int operator() (const PTDType& ptd, int i) const noexcept {
        Box tbx;
        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
        auto tid = getTileIndex(iv, box, true, bin_size, tbx);
        if (bin_idx != -1) {
            auto group = ptd.m_idata[bin_idx][i];
            return static_cast<unsigned int>(tid * max_group + group);
        } else {
            return static_cast<unsigned int>(tid);
        }
    }

private:

    GpuArray<Real,AMREX_SPACEDIM> plo;
    GpuArray<Real,AMREX_SPACEDIM> dxi;
    Box domain;
    IntVect bin_size;
    Box box;
    int max_group;
    int bin_idx;
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
template <typename AgentContainer, typename PTDType, typename CandidateFunc, typename BinaryInteractionFunc>
void interactAgentsImpl(AgentContainer& agents, int bin_idx)
{
    BL_PROFILE("interactAgentsimpl");
    int n_disease = agents.numDiseases();
    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    CandidateFunc isCandidate;
    BinaryInteractionFunc binaryInteraction;

    for (int lev = 0; lev < agents.numLevels(); ++lev)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto num_tiles = numTilesInBox(mfi.validbox(), true, bin_size);

            int max_group = 0;
            if (bin_idx != -1) {
                auto bingroup_ptr = soa.GetIntData(bin_idx).data();
                for (int i = 0; i < np; i++) {
                    max_group = max(max_group, bingroup_ptr[i]);
                }
            }
            max_group++;

            // agents are binned by community + another group. These values can change from one time step to the next so they
            // have to be recomputed (rebuild the bins) every time step
            Binner<PTDType> binner(agents.Geom(lev), bin_size, mfi.validbox(), max_group, bin_idx);
            DenseBins<PTDType> bins;
            bins.build(BinPolicy::GPU, np, ptd, num_tiles * max_group, binner);

            AMREX_ALWAYS_ASSERT(bins.numBins() >= 0);
            auto inds = bins.permutationPtr();
            auto offsets = bins.offsetsPtr();

            for (int d = 0; d < n_disease; d++) {
                int r_RT = RealIdx::nattribs;
                auto prob_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::prob).data();
                auto lparm = agents.getDiseaseParameters_d(d);
                Real scale = 1.0_prt;  // TODO this should vary based on cell
                Real infect = lparm->infect;
                infect *= (1.0_rt - lparm->vac_eff);

                ParallelFor(bins.numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept {
                    auto infectious_i = inds[ii];
                    int i_cell = binner(ptd, infectious_i);
                    auto cell_start = offsets[i_cell];
                    auto cell_stop = offsets[i_cell + 1];
                    AMREX_ALWAYS_ASSERT((Long)infectious_i < np);
                    // Doing infectious first requires an atomic operation for GPUs, but generally requires far fewer operations
                    // because there are usually more susceptible agents than infectious. This can be a large performance
                    // difference for CPU only runs.
                    if (isInfectious(infectious_i, ptd, d) && isCandidate(infectious_i, ptd)) {
                        //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                        for (auto jj = cell_start; jj < cell_stop; ++jj) {
                            auto susceptible_i = inds[jj];
                            AMREX_ALWAYS_ASSERT((Long)susceptible_i < np);
                            if (infectious_i != susceptible_i &&
                                isSusceptible(susceptible_i, ptd, d) &&
                                isCandidate(susceptible_i, ptd)) {
                                auto prob = binaryInteraction(infectious_i, susceptible_i, ptd, lparm, scale);
                                // The atomic operation is needed because we find all the susceptible for each infectious in turn.
                                // It can be eliminated by switching the order of infectious and susceptible.
                                Gpu::Atomic::Multiply(&prob_ptr[susceptible_i], prob);
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
