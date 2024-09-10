/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;


/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at school. */
template <typename PTDType>
struct BinaryInteractionSchool {
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    ParticleReal operator() (const int infectious_i, /*!< Index of infectious agent */
                             const int susceptible_i, /*!< Index of susceptible agent */
                             const PTDType& a_ptd, /*!< Particle tile data */
                             const DiseaseParm* const a_lparm, /*!< disease paramters */
                             const Real a_social_scale  /*!< Social scale */) const noexcept {
        Real infect = a_lparm->infect;
        infect *= (1.0_rt - a_lparm->vac_eff);

        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::school][infectious_i] == a_ptd.m_idata[IntIdx::school][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);

        auto age_group_ptr = a_ptd.m_idata[IntIdx::age_group];
        auto nborhood_ptr = a_ptd.m_idata[IntIdx::nborhood];
        auto school_ptr = a_ptd.m_idata[IntIdx::school];

        // binned so that infectious and susceptible are in the same school
        int school = school_ptr[susceptible_i];
        //infect *= i_mask;
        //infect *= j_mask;
        ParticleReal prob = 1.0_prt;
        if (nborhood_ptr[infectious_i] == nborhood_ptr[susceptible_i]) {
            if (school > 5) {             // Playgroup
                prob *= 1.0_prt - infect * a_lparm->xmit_school[6] * a_social_scale;
            } else if (school == 5) {     // Day care
                prob *= 1.0_prt - infect * a_lparm->xmit_school[5] * a_social_scale;
            }
        }
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
        }
        return prob;
    }
};


template <typename PTDType>
struct SchoolCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) &&
                ptd.m_idata[IntIdx::school][idx] > 0 &&
                !ptd.m_idata[IntIdx::withdrawn][idx] &&
                ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};



/*! \brief Class describing agent interactions at school */
template <typename PCType, typename PTDType, typename PType>
class InteractionModSchool : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModSchool() { }

        /*! \brief default destructor */
        virtual ~InteractionModSchool() = default;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
            interactAgentsImpl<PCType, PTDType, SchoolCandidate<PTDType>, BinaryInteractionSchool<PTDType>>(agents, IntIdx::school);
        }

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents (PCType&, MultiFab&, PCType&) override {
            amrex::Abort("Do not use this interface for this interaction model");
        }

    protected:

    private:
};


#endif
