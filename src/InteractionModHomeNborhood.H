/*! @file InteractionModHomeNborhood.H
 * \brief Contains the class describing agent interactions at neighborhood
 */

#ifndef _INTERACTION_MOD_HOME_NBORHOOD_H_
#define _INTERACTION_MOD_HOME_NBORHOOD_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent in the neighborhood/community. */
template <typename PTDType>
struct BinaryInteractionHomeNborhood {
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    ParticleReal operator() (const int infectious_i, /*!< Index of infectious agent */
                             const int susceptible_i, /*!< Index of susceptible agent */
                             const PTDType& a_ptd, /*!< Particle tile data */
                             const DiseaseParm* const a_lparm, /*!< disease paramters */
                             const Real a_social_scale /*!< Social scale */) const noexcept {
        Real infect = a_lparm->infect;
        infect *= (1.0_rt - a_lparm->vac_eff);

        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
        auto school_ptr = a_ptd.m_idata[IntIdx::school];
        auto random_travel_ptr = a_ptd.m_idata[IntIdx::random_travel];

        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::home_i][infectious_i] == a_ptd.m_idata[IntIdx::home_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::home_j][infectious_i] == a_ptd.m_idata[IntIdx::home_j][susceptible_i]);

        //infect *= i_mask;
        //infect *= j_mask;
        ParticleReal prob = 1.0_prt;
        // define neighbordhood based on whether agent is in home or work community
        int nborhood_infectious_i = nborhood_ptr[infectious_i];
        int nborhood_susceptible_i = nborhood_ptr[susceptible_i];

        // school < 0 means a child normally attends school, but not today
        // Should always be in the same community = same cell
        // Neighborhood?
        if (nborhood_infectious_i == nborhood_susceptible_i) {
            if (school_ptr[infectious_i] < 0) { // not attending school, use _SC contacts
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
        return prob;
    }
};

template <typename PTDType>
struct HomeNborhoodCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        // this is the only case where we allow random travelers to interact
        return !isHospitalized(idx, ptd) && !ptd.m_idata[IntIdx::withdrawn][idx];
    }
};



/*! \brief Class describing agent interactions in the neighborhood/community */
template <typename PCType, typename PTDType, typename PType>
class InteractionModHomeNborhood : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModHomeNborhood() { }

        /*! \brief default destructor */
        virtual ~InteractionModHomeNborhood() = default;

        /*! \brief Simulate agent interaction in the neighborhood/community */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
            // passing -1 for the binning group indicates bin only by community (cell)
            interactAgentsImpl<InteractionModHomeNborhood<PCType, PTDType, PType>, PCType, PTDType,
                               HomeNborhoodCandidate<PTDType>,
                               BinaryInteractionHomeNborhood<PTDType>>(*this, agents, -1);
        }

    protected:

    private:
};


#endif
