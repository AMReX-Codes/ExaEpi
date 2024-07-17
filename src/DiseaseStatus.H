/*! @file DiseaseStatus.H
 *  \brief Defines class for disease status updates */

#ifndef _DISEASE_STATUS_H_
#define _DISEASE_STATUS_H_

#include <vector>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>

#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief Disease status and its updates for each agent
 *
 *  Contains data and functions for updating disease status for agents.
 */
template <  typename AC /*!< agent container type */,
            typename ACT, /*!< agent container tile type */
            typename ACTD, /*!< agent container tile data type */
            typename A /*!< agent type */>
class DiseaseStatus
{
    public:

        /*! \brief constructor */
        DiseaseStatus<AC,ACT,ACTD,A>();

        /*! \brief default destructor */
        virtual ~DiseaseStatus<AC,ACT,ACTD,A>() = default;

        /*! \brief Updates disease status of each agent at a given step and
         *   also updates a MultiFab that tracks disease statistics (hospitalization,
         *   ICU, ventilator, and death) in a community. */
        virtual void updateAgents(AC&, MultiFab&, const int) const;

    protected:

        /*! sick -> hospital probabilities */
        Vector<Real> m_CHR = {.0104_rt, .0104_rt, .070_rt, .28_rt, 1.0_rt};
        /*! hospital -> ICU probabilities */
        Vector<Real> m_CIC = {.24_rt, .24_rt, .24_rt, .36_rt, .35_rt};
        /*! ICU -> ventilator probabilities */
        Vector<Real> m_CVE = {.12_rt, .12_rt, .12_rt, .22_rt, .22_rt};
        /*! ventilator -> dead probilities */
        Vector<Real> m_CVF = {.20_rt, .20_rt, .20_rt, 0.45_rt, 1.26_rt};

        /*! sick -> hospital probabilities */
        Gpu::DeviceVector<Real> m_CHR_d;
        /*! hospital -> ICU probabilities */
        Gpu::DeviceVector<Real> m_CIC_d;
        /*! ICU -> ventilator probabilities */
        Gpu::DeviceVector<Real> m_CVE_d;
        /*! ventilator -> dead probilities */
        Gpu::DeviceVector<Real> m_CVF_d;

        /*! \brief Copy host vector to device */
        inline void copyToDevice (  const Vector<Real>&      a_vec_h, /*!< host vector */
                                    Gpu::DeviceVector<Real>& a_vec_d  /*!< device vector */)
        {
            a_vec_d.resize(0);
            a_vec_d.resize(a_vec_h.size());
            Gpu::copyAsync( Gpu::hostToDevice,
                            a_vec_h.begin(),
                            a_vec_h.end(),
                            a_vec_d.begin() );
        }
};

/*! Constructor */
template<typename AC, typename ACT, typename ACTD, typename A>
DiseaseStatus<AC,ACT,ACTD,A>::DiseaseStatus()
{
    copyToDevice(m_CHR, m_CHR_d);
    copyToDevice(m_CIC, m_CIC_d);
    copyToDevice(m_CVE, m_CVE_d);
    copyToDevice(m_CVF, m_CVF_d);
    Gpu::streamSynchronize();
}

/*! At a given step, update the disease status of each agent based on the following overall logic:
    + If agent status is #Status::never or #Status::susceptible, do nothing
    + If agent status is #Status::infected, then
      + Increment its counter by 1 day
      + If counter is within incubation period, do nothing more
      + Else, use hospitalization probabilities (by age group)
        to decide if agent is hospitalized. If yes, use age group to set hospital timer. Also, use
        age-group-wise probabilities to move agent to ICU and then to ventilator. Adjust timer
        accordingly.
      + Update the community-wise disease stats tracker MultiFab according to hospitalization/ICU/vent
        status (using the agent's home community)
      + Else (beyond 3 days), count down hospital timer if agent is hospitalized. At end of hospital
        stay, determine if agent is #Status dead or #Status::immune. For non-hospitalized agents,
        set them to #Status::immune after incubation length + infection length days.

    The input argument is a MultiFab with 4 components corresponding to "hospitalizations", "ICU",
    "ventilator", and "death". It contains the cumulative totals of these quantities for each
    community as the simulation progresses.
*/
template<typename AC, typename ACT, typename ACTD, typename A>
void DiseaseStatus<AC,ACT,ACTD,A>::updateAgents(AC& a_agents, /*!< Agent containter */
                                                MultiFab& a_stats, /*!< MultiFab to store disease stats */
                                                const int a_idx /*!< disease index */ ) const
{
    BL_PROFILE("DiseaseStatus::updateAgents");
    AMREX_ASSERT(a_idx < a_agents.numDiseases());

    for (int lev = 0; lev <= a_agents.finestLevel(); ++lev)
    {
        auto& plev  = a_agents.GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto timer_ptr = soa.GetRealData(RealIdx::treatment_timer).data();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;

            auto status_ptr      = soa.GetIntData(i_RT+i0(a_idx)+IntIdxDisease::status).data();
            auto symptomatic_ptr = soa.GetIntData(i_RT+i0(a_idx)+IntIdxDisease::symptomatic).data();

            auto counter_ptr           = soa.GetRealData(r_RT+r0(a_idx)+RealIdxDisease::disease_counter).data();
            auto prob_ptr              = soa.GetRealData(r_RT+r0(a_idx)+RealIdxDisease::prob).data();
            auto incubation_period_ptr = soa.GetRealData(r_RT+r0(a_idx)+RealIdxDisease::incubation_period).data();
            auto infectious_period_ptr = soa.GetRealData(r_RT+r0(a_idx)+RealIdxDisease::infectious_period).data();
            auto symptomdev_period_ptr = soa.GetRealData(r_RT+r0(a_idx)+RealIdxDisease::symptomdev_period).data();

            auto* disease_parm_d = a_agents.getDiseaseParameters_d(a_idx);
            auto* disease_parm_h = a_agents.getDiseaseParameters_h(a_idx);
            auto mean_immune_time = disease_parm_h->mean_immune_time;
            auto immune_time_spread = disease_parm_h->immune_time_spread;

            auto ds_arr = a_stats[mfi].array();

            auto symptomatic_withdraw = a_agents.symptomaticWithdraw();
            auto symptomatic_withdraw_compliance = a_agents.symptomaticWithdrawCompliance();

            auto CHR = m_CHR_d.data();
            auto CIC = m_CIC_d.data();
            auto CVE = m_CVE_d.data();
            auto CVF = m_CVF_d.data();

            ParallelForRNG( np,
                            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                prob_ptr[i] = 1.0_rt;
                if ( status_ptr[i] == Status::never ||
                     status_ptr[i] == Status::susceptible ) {
                    return;
                }
                else if (status_ptr[i] == Status::immune) {
                    counter_ptr[i] -= 1.0_rt;
                    if (counter_ptr[i] < 0.0_rt) {
                        counter_ptr[i] = 0.0_rt;
                        timer_ptr[i] = 0.0_rt;
                        status_ptr[i] = Status::susceptible;
                        return;
                    }
                }
                else if (status_ptr[i] == Status::infected) {
                    counter_ptr[i] += 1;
                    if (counter_ptr[i] == 1) {
                        if (Random(engine) < disease_parm_d->p_asymp[0]) {
                            symptomatic_ptr[i] = SymptomStatus::asymptomatic;
                        } else {
                            symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                        }
                    }
                    if (counter_ptr[i] == Math::floor(symptomdev_period_ptr[i])) {
                        if (symptomatic_ptr[i] != SymptomStatus::asymptomatic) {
                            symptomatic_ptr[i] = SymptomStatus::symptomatic;
                        }
                        if (    (symptomatic_ptr[i] == SymptomStatus::symptomatic)
                            &&  (symptomatic_withdraw)
                            &&  (Random(engine) < symptomatic_withdraw_compliance)) {
                            withdrawn_ptr[i] = 1;
                        }
                    }
                    if (counter_ptr[i] < incubation_period_ptr[i]) {
                        // incubation phase
                        return;
                    }
                    if (counter_ptr[i] == Math::ceil(incubation_period_ptr[i])) {
                        // decide if hospitalized
                        Real p_hosp = CHR[age_group_ptr[i]];
                        if (Random(engine) < p_hosp) {
                            if ((age_group_ptr[i]) < 3) {  // age groups 0-4, 5-18, 19-29
                                timer_ptr[i] = 3;  // Ages 0-49 hospitalized for 3.1 days
                            }
                            else if (age_group_ptr[i] == 4) {
                                timer_ptr[i] = 7;  // Age 65+ hospitalized for 6.5 days
                            }
                            else if (Random(engine) < 0.57) {
                                timer_ptr[i] = 3;  // Proportion of 30-64 that is under 50
                            }
                            else {
                                timer_ptr[i] = 8;  // Age 50-64 hospitalized for 7.8 days
                            }
                            Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                    home_j_ptr[i],
                                                                    0,
                                                                    DiseaseStats::hospitalization ),
                                                          1.0_rt );
                            if (Random(engine) < CIC[age_group_ptr[i]]) {
                                timer_ptr[i] += 10;  // move to ICU
                                Gpu::Atomic::AddNoRet( &ds_arr( home_i_ptr[i],
                                                                       home_j_ptr[i],
                                                                       0,
                                                                       DiseaseStats::ICU ),
                                                              1.0_rt );
                                if (Random(engine) < CVE[age_group_ptr[i]]) {
                                    Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                     home_j_ptr[i],
                                                                     0,
                                                                     DiseaseStats::ventilator ),
                                                           1.0_rt );
                                    timer_ptr[i] += 10;  // put on ventilator
                                }
                            }
                        }
                    } else {
                        if (timer_ptr[i] > 0.0_rt) {
                            // do hospital things
                            timer_ptr[i] -= 1.0_rt;
                            if (timer_ptr[i] == 0) {
                                if (CVF[age_group_ptr[i]] > 2.0_rt) {
                                    if (Random(engine) < (CVF[age_group_ptr[i]] - 2.0_rt)) {
                                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                         home_j_ptr[i],
                                                                         0,
                                                                         DiseaseStats::death),
                                                               1.0_rt );
                                        status_ptr[i] = Status::dead;
                                    }
                                }
                                Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                 home_j_ptr[i],
                                                                 0,
                                                                 DiseaseStats::hospitalization ),
                                                       -1.0_rt );
                                if (status_ptr[i] != Status::dead) {
                                    status_ptr[i] = Status::immune;  // If alive, hospitalized patient recovers
                                    counter_ptr[i] = (mean_immune_time - immune_time_spread)
                                                     + 2.0_rt*immune_time_spread*Random(engine);
                                    symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                    withdrawn_ptr[i] = 0;
                                }
                            }
                            if (timer_ptr[i] == 10) {
                                if (CVF[age_group_ptr[i]] > 1.0_rt) {
                                    if (Random(engine) < (CVF[age_group_ptr[i]] - 1.0_rt)) {
                                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                         home_j_ptr[i],
                                                                         0,
                                                                         DiseaseStats::death ),
                                                               1.0_rt);
                                        status_ptr[i] = Status::dead;
                                    }
                                }
                                Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                 home_j_ptr[i],
                                                                 0,
                                                                 DiseaseStats::hospitalization ),
                                                       -1.0_rt );
                                Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                 home_j_ptr[i],
                                                                 0,
                                                                 DiseaseStats::ICU),
                                                       -1.0_rt );
                                if (status_ptr[i] != Status::dead) {
                                    status_ptr[i] = Status::immune;  // If alive, ICU patient recovers
                                    counter_ptr[i] = (mean_immune_time - immune_time_spread)
                                                    + 2.0_rt*immune_time_spread*Random(engine);
                                    symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                    withdrawn_ptr[i] = 0;
                                }
                            }
                            if (timer_ptr[i] == 20) {
                                if (Random(engine) < CVF[age_group_ptr[i]]) {
                                    Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                     home_j_ptr[i],
                                                                     0,
                                                                     DiseaseStats::death ),
                                                           1.0_rt );
                                    status_ptr[i] = Status::dead;
                                }
                                Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                 home_j_ptr[i],
                                                                 0,
                                                                 DiseaseStats::hospitalization ),
                                                       -1.0_rt );
                                Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                 home_j_ptr[i],
                                                                 0,
                                                                 DiseaseStats::ICU ),
                                                       -1.0_rt );
                                Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                                 home_j_ptr[i],
                                                                 0,
                                                                 DiseaseStats::ventilator ),
                                                       -1.0_rt );
                                if (status_ptr[i] != Status::dead) {
                                    status_ptr[i] = Status::immune;  // If alive, ventilated patient recovers
                                    counter_ptr[i] = (mean_immune_time - immune_time_spread)
                                                     + 2.0_rt*immune_time_spread*Random(engine);
                                    symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                    withdrawn_ptr[i] = 0;
                                }
                            }
                        } else { // not hospitalized, recover once not infectious
                            if (counter_ptr[i] >= (incubation_period_ptr[i] + infectious_period_ptr[i])) {
                                status_ptr[i] = Status::immune;
                                counter_ptr[i] = (mean_immune_time - immune_time_spread)
                                                 + 2.0_rt*immune_time_spread*Random(engine);
                                symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                withdrawn_ptr[i] = 0;
                            }
                        }
                    }
                }
            });
        }
    }
}
#endif
