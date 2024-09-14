/*! @file InteractionModWorkNborhood.H
 * \brief Contains the class describing agent interactions at neighborhood
 */

#ifndef _INTERACTION_MOD_WORK_NBORHOOD_H_
#define _INTERACTION_MOD_WORK_NBORHOOD_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent in the neighborhood/community. */
template <typename PTDType>
struct BinaryInteractionWorkNborhood {
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    ParticleReal operator() (const int infectious_i, /*!< Index of infectious agent */
                             const int susceptible_i, /*!< Index of susceptible agent */
                             const PTDType& a_ptd, /*!< Particle tile data */
                             const DiseaseParm* const a_lparm, /*!< disease paramters */
                             const Real a_social_scale /*!< Social scale */) const noexcept {
        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
        auto work_nborhood_ptr = a_ptd.m_idata[IntIdx::work_nborhood];
        auto random_travel_ptr = a_ptd.m_idata[IntIdx::random_travel];

        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);
        AMREX_ALWAYS_ASSERT(random_travel_ptr[infectious_i] < 0 && random_travel_ptr[susceptible_i] < 0);

        //infect *= i_mask;
        //infect *= j_mask;
        // define neighbordhood based on age of agent
        int nborhood_infectious = age_group_ptr[infectious_i] > 1 ? work_nborhood_ptr[infectious_i] : nborhood_ptr[infectious_i];
        int nborhood_susceptible = age_group_ptr[susceptible_i] > 1 ? work_nborhood_ptr[susceptible_i] : nborhood_ptr[susceptible_i];

        // school < 0 means a child normally attends school, but not today
        // Should always be in the same community = same cell
        // Neighborhood?
        if (nborhood_infectious == nborhood_susceptible) {
            return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_hood_SC, a_lparm->xmit_hood) * a_social_scale;
        } else { // Community?
            return infectProb(a_ptd, infectious_i, susceptible_i, a_lparm->xmit_comm_SC, a_lparm->xmit_comm) * a_social_scale;
        }
    }
};

template <typename PTDType>
struct WorkNborhoodCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) && !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};


/*! \brief Class describing agent interactions in the neighborhood/community */
template <typename PCType, typename PTDType, typename PType>
class InteractionModWorkNborhood : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModWorkNborhood (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModWorkNborhood() = default;

        /*! \brief Simulate agent interaction in the neighborhood/community */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
            // passing -1 for the binning group indicates bin only by community (cell)
            interactAgentsImpl<InteractionModWorkNborhood<PCType, PTDType, PType>, PCType, PTDType,
                                WorkNborhoodCandidate<PTDType>,
                                BinaryInteractionWorkNborhood<PTDType>>(*this, agents, -1);
        }

    protected:

    private:
};


#endif
