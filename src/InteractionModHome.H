/*! @file InteractionModHome.H
 * \brief Contains the class describing agent interactions at home
 */

#ifndef _INTERACTION_MOD_HOME_H_
#define _INTERACTION_MOD_HOME_H_

#include "InteractionModel.H"
#include "DiseaseParm.H"
#include "AgentDefinitions.H"

using namespace amrex;

#define FAMILIES_PER_CLUSTER 4

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at home. */
template <typename PTDType>
struct BinaryInteractionHome {
    AMREX_GPU_DEVICE
    ParticleReal operator() (const int infectious_i, /*!< Index of infectious agent */
                             const int susceptible_i, /*!< Index of susceptible agent */
                             const PTDType& a_ptd, /*!< Particle tile data */
                             const DiseaseParm* const a_lparm,  /*!< disease paramters */
                             const Real a_social_scale /*!< Social scale */) const noexcept {
        Real infect = a_lparm->infect;
        infect *= (1.0_rt - a_lparm->vac_eff);

        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto family_ptr = a_ptd.m_idata[IntIdx::family];
        auto school_ptr = a_ptd.m_idata[IntIdx::school];
        auto withdrawn_ptr = a_ptd.m_idata[IntIdx::withdrawn];

        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::home_i][infectious_i] == a_ptd.m_idata[IntIdx::home_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::home_j][infectious_i] == a_ptd.m_idata[IntIdx::home_j][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::nborhood][infectious_i] == a_ptd.m_idata[IntIdx::nborhood][susceptible_i]);

        //infect *= i_mask;
        //infect *= j_mask;
        if (family_ptr[infectious_i] == family_ptr[susceptible_i]) {
            // at home, within a family
            if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
                return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_child_SC, a_lparm->xmit_child, infect);
            } else {
                return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_adult_SC, a_lparm->xmit_adult, infect);
            }
        } else if (family_ptr[infectious_i] / FAMILIES_PER_CLUSTER == family_ptr[susceptible_i] / FAMILIES_PER_CLUSTER &&
                   !withdrawn_ptr[infectious_i] && !withdrawn_ptr[susceptible_i]) {
            infect *= a_social_scale;
            // family cluster
            if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
                return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_nc_child_SC, a_lparm->xmit_nc_child, infect);
            } else {
                return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_nc_adult_SC, a_lparm->xmit_nc_adult, infect);
            }
        }
        return 1.0_prt;
    }
};

template <typename PTDType>
struct HomeCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) && ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Class describing agent interactions at home */
template <typename PCType, typename PTDType, typename PType>
class InteractionModHome : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModHome(bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModHome() = default;

        /*! \brief Simulate agent interaction at home */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
            interactAgentsImpl<InteractionModHome<PCType, PTDType, PType>, PCType, PTDType,
                               HomeCandidate<PTDType>, BinaryInteractionHome<PTDType>>(*this, agents, IntIdx::nborhood);
        }

    protected:

    private:
};


#endif
