/*! @file InteractionModWork.H
 * \brief Contains the class describing agent interactions at work
 */

#ifndef _INTERACTION_MOD_WORK_H_
#define _INTERACTION_MOD_WORK_H_

#include "InteractionModel.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent at work. */
template <typename PTDType>
struct BinaryInteractionWork {
    AMREX_GPU_HOST_DEVICE
    ParticleReal operator() (const int infectious_i,
                             const int susceptible_i,
                             const PTDType& a_ptd,
                             const DiseaseParm* const a_lparm,
                             const Real a_work_scale) const noexcept {
        Real infect = a_lparm->infect;
        infect *= (1.0_rt - a_lparm->vac_eff);

        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::workgroup][infectious_i] == a_ptd.m_idata[IntIdx::workgroup][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);
        //infect *= i_mask;
        //infect *= j_mask;
        ParticleReal prob = 1.0_prt - infect * a_lparm->xmit_work * a_work_scale;
        return prob;
    }
};

template <typename PTDType>
struct WorkCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator() (const int idx, const PTDType& ptd) const noexcept {
        return !isHospitalized(idx, ptd) &&
                ptd.m_idata[IntIdx::work_i][idx] >= 0 &&
                ptd.m_idata[IntIdx::workgroup][idx] > 0 &&
                !ptd.m_idata[IntIdx::withdrawn][idx] &&
                ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};


/*! \brief Class describing agent interactions at work */
template <typename PCType, typename PTDType, typename PType>
class InteractionModWork : public InteractionModel<PCType, PTDType, PType>
{
    public:

        /*! \brief null constructor */
        InteractionModWork () {}

        /*! \brief default destructor */
        virtual ~InteractionModWork () = default;

        /*! \brief Simulate agent interaction at work */
        virtual void interactAgents (PCType& agents, MultiFab&) override {
            interactAgentsImpl<InteractionModWork<PCType, PTDType, PType>, PCType, PTDType,
                               WorkCandidate<PTDType>,
                               BinaryInteractionWork<PTDType>>(*this, agents, IntIdx::workgroup);
        }

    protected:

    private:
};


#endif
