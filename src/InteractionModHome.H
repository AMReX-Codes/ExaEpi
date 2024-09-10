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
        ParticleReal prob = 1.0_prt;
        if (family_ptr[infectious_i] == family_ptr[susceptible_i]) {
            // at home, within a family
            if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
                if (school_ptr[infectious_i] < 0) { // not attending school, use _SC contacts
                    prob *= 1.0_prt - infect * a_lparm->xmit_child_SC[age_group_ptr[susceptible_i]];
                } else {
                    prob *= 1.0_prt - infect * a_lparm->xmit_child[age_group_ptr[susceptible_i]];
                }
            } else {
                if (school_ptr[infectious_i] < 0) { // not attending school, use _SC contacts
                    prob *= 1.0_prt - infect * a_lparm->xmit_adult_SC[age_group_ptr[susceptible_i]];
                } else {
                    prob *= 1.0_prt - infect * a_lparm->xmit_adult[age_group_ptr[susceptible_i]];
                }
            }
        } else if (family_ptr[infectious_i] / FAMILIES_PER_CLUSTER == family_ptr[susceptible_i] / FAMILIES_PER_CLUSTER &&
                   !withdrawn_ptr[infectious_i] && !withdrawn_ptr[susceptible_i]) {
            // family cluster
            if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
                if (school_ptr[infectious_i] < 0)  // not attending school, use _SC contacts
                    prob *= 1.0_prt - infect * a_lparm->xmit_nc_child_SC[age_group_ptr[susceptible_i]] * a_social_scale;
                else
                    prob *= 1.0_prt - infect * a_lparm->xmit_nc_child[age_group_ptr[susceptible_i]] * a_social_scale;
            } else {
                if (school_ptr[infectious_i] < 0) { // not attending school, use _SC contacts
                    prob *= 1.0_prt - infect * a_lparm->xmit_nc_adult_SC[age_group_ptr[susceptible_i]] * a_social_scale;
                } else {
                    prob *= 1.0_prt - infect * a_lparm->xmit_nc_adult[age_group_ptr[susceptible_i]] * a_social_scale;
                }
            }
        }
        return prob;
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
        InteractionModHome() { }

        /*! \brief default destructor */
        virtual ~InteractionModHome() = default;

        /*! \brief Simulate agent interaction at home */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
            interactAgentsImpl<PCType, PTDType, HomeCandidate<PTDType>, BinaryInteractionHome<PTDType>>(agents, IntIdx::nborhood);
        }

        /*! \brief Simulate agent interaction at home */
        virtual void interactAgents (PCType&, MultiFab&, PCType&) override {
            amrex::Abort("Do not use this interface for this interaction model");
        }

    protected:

    private:
};


#endif
