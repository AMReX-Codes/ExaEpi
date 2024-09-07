/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


template <typename PTDType>
struct SchoolBinner
{
    SchoolBinner(const Geometry &geom, const IntVect &bin_size, const Box &box, const int max_school, const int max_work_nborhood) :
                 bin_size(bin_size), box(box), max_school(max_school), max_work_nborhood(max_work_nborhood) {
        dxi = geom.InvCellSizeArray();
        plo = geom.ProbLoArray();
        domain = geom.Domain();
    }

    AMREX_GPU_HOST_DEVICE
    unsigned int operator() (const PTDType& ptd, int i) const noexcept {
        Box tbx;
        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
        auto tid = getTileIndex(iv, box, true, bin_size, tbx);
        auto school = ptd.m_idata[IntIdx::school][i];
        auto work_nborhood = ptd.m_idata[IntIdx::work_nborhood][i];
        return static_cast<unsigned int>((tid * max_school + school) * max_work_nborhood + work_nborhood);
    }

  private:

    GpuArray<Real,AMREX_SPACEDIM> plo;
    GpuArray<Real,AMREX_SPACEDIM> dxi;
    Box domain;
    IntVect bin_size;
    Box box;
    int max_school;
    int max_work_nborhood;
};

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at school. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionSchool (const int infectious_i, /*!< Index of infectious agent */
                                     const int susceptible_i, /*!< Index of susceptible agent */
                                     const PTDType& a_ptd, /*!< Particle tile data */
                                     const DiseaseParm* const a_lparm, /*!< disease paramters */
                                     const Real a_social_scale,  /*!< Social scale */
                                     ParticleReal* const a_prob_ptr /*!< infection probability */)
{
    Real infect = a_lparm->infect;
    infect *= (1.0_rt - a_lparm->vac_eff);

    AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::school][infectious_i] == a_ptd.m_idata[IntIdx::school][susceptible_i]);
    AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_nborhood][infectious_i] == a_ptd.m_idata[IntIdx::work_nborhood][susceptible_i]);
    AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                        a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);

    auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
    auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
    auto school_ptr = a_ptd.m_idata[IntIdx::school];

    // binned so that infectious and susceptible are in the same school
    int school = school_ptr[infectious_i];
    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0_prt;
    // Elementary/middle/high school in common
    if (school > 0 && school < 5) {
        if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
            if (age_group_ptr[susceptible_i] <= 1) {  // Receiver j is a child
                prob *= 1.0_prt - infect * a_lparm->xmit_school[school] * a_social_scale;
            } else {  // Child student -> adult teacher/staff transmission
                prob  *= 1.0_prt - infect * a_lparm->xmit_sch_c2a[school] * a_social_scale;
            }
        } else if (age_group_ptr[susceptible_i] <= 1) {  // Adult teacher/staff -> child student
            prob *= 1.0_prt - infect * a_lparm->xmit_sch_a2c[school] * a_social_scale;
        }
    } else {
        if (nborhood_ptr[infectious_i] == nborhood_ptr[susceptible_i]) {
            if (school > 5) {             // Playgroup
                prob *= 1.0_prt - infect * a_lparm->xmit_school[6] * a_social_scale;
            } else if (school == 5) {     // Day care
                prob *= 1.0_prt - infect * a_lparm->xmit_school[5] * a_social_scale;
            }
        }
    }
    Gpu::Atomic::Multiply(&a_prob_ptr[susceptible_i], prob);
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
        virtual void interactAgents( AC&, MultiFab& ) override;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents( AC&, MultiFab&, AC& ) override {
            amrex::Abort("Do not use this interface for this interaction model");
        }

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
            auto school_ptr = soa.GetIntData(IntIdx::school).data();
            auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();

            int max_school = 0;
            int max_work_nborhood = 0;
            for (int i = 0; i < np; i++) {
                max_school = max(max_school, school_ptr[i]);
                max_work_nborhood = max(max_work_nborhood, work_nborhood_ptr[i]);
            }
            max_school++;
            max_work_nborhood++;

            // agents are binned by community + workgroup. These values can change from one time step to the next so they
            // have to be recomputed (rebuild the bins) every time step
            SchoolBinner<ACTD> binner(a_agents.Geom(lev), bin_size, mfi.validbox(), max_school, max_work_nborhood);
            DenseBins<ACTD> bins_school;
            bins_school.build(BinPolicy::GPU, np, ptd, num_tiles * max_school * max_work_nborhood, binner);

            AMREX_ALWAYS_ASSERT(bins_school.numBins() >= 0);
            auto inds = bins_school.permutationPtr();
            auto offsets = bins_school.offsetsPtr();

            for (int d = 0; d < n_disease; d++) {
                auto prob_ptr = this->getAgentProbPtr(a_agents, lev, mfi, d);
                //auto mask_arr = a_mask[mfi].array();
                auto lparm = a_agents.getDiseaseParameters_d(d);
                Real social_scale = 1.0_prt;  // TODO this should vary based on cell
                Real infect = lparm->infect;
                infect *= (1.0_rt - lparm->vac_eff);

                ParallelFor(bins_school.numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept {
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
                        school_ptr[infectious_i] > 0 &&        // at school
                        !withdrawn_ptr[infectious_i] &&        // not withdrawn
                        random_travel_ptr[infectious_i] < 0)   // not traveling
                    {
                        //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                        for (auto jj = cell_start; jj < cell_stop; ++jj) {
                            auto susceptible_i = inds[jj];
                            AMREX_ALWAYS_ASSERT((Long)susceptible_i < np);
                            if (infectious_i != susceptible_i &&
                                isSusceptible(susceptible_i, ptd, d) &&
                                !isHospitalized(susceptible_i, ptd) &&
                                work_nborhood_ptr[infectious_i] == work_nborhood_ptr[susceptible_i] &&
                                school_ptr[susceptible_i] > 0 &&
                                !withdrawn_ptr[susceptible_i] &&
                                random_travel_ptr[susceptible_i] < 0)
                            {
                                binaryInteractionSchool(infectious_i, susceptible_i, ptd, lparm, social_scale, prob_ptr);
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
