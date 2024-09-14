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
        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
        auto school_ptr = a_ptd.m_idata[IntIdx::school];

        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::home_i][infectious_i] == a_ptd.m_idata[IntIdx::home_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::home_j][infectious_i] == a_ptd.m_idata[IntIdx::home_j][susceptible_i]);

        //infect *= i_mask;
        //infect *= j_mask;

        // Should always be in the same community = same cell
        // Neighborhood?
        if (nborhood_ptr[infectious_i] == nborhood_ptr[susceptible_i]) {
            return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_hood_SC, a_lparm->xmit_hood) * a_social_scale;
        } else { // Community?
            return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_comm_SC, a_lparm->xmit_comm) * a_social_scale;
        }
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
        InteractionModHomeNborhood (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModHomeNborhood () = default;

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
