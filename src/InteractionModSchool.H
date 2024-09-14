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
        // Elementary/middle/high school in common
        if (school > 0 && school < 5) {
            if (age_group_ptr[infectious_i] <= 1) {  // Transmitter i is a child
                if (age_group_ptr[susceptible_i] <= 1) {  // Receiver j is a child
                    return a_lparm->xmit_school[school] * a_social_scale;
                } else {  // Child student -> adult teacher/staff transmission
                    return a_lparm->xmit_sch_c2a[school] * a_social_scale;
                }
            } else if (age_group_ptr[susceptible_i] <= 1) {  // Adult teacher/staff -> child student
                return a_lparm->xmit_sch_a2c[school] * a_social_scale;
            }
        } else {
            if (nborhood_ptr[infectious_i] == nborhood_ptr[susceptible_i]) {
                if (school > 5) {             // Playgroup
                    return a_lparm->xmit_school[6] * a_social_scale;
                } else if (school == 5) {     // Day care
                    return a_lparm->xmit_school[5] * a_social_scale;
                }
            }
        }
        return 0.0_prt;
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
        InteractionModSchool (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModSchool () = default;

        /*! \brief Simulate agent interaction at school */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
            interactAgentsImpl<InteractionModSchool<PCType, PTDType, PType>, PCType, PTDType,
                               SchoolCandidate<PTDType>,
                               BinaryInteractionSchool<PTDType>>(*this, agents, IntIdx::school);
        }

    protected:

    private:
};


#endif
