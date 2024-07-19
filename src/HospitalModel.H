/*! @file HospitalModel.H
 * \brief Contains the class describing agent interactions at a hospital
 */

#ifndef _HOSPITAL_MODEL_H_
#define _HOSPITAL_MODEL_H_

#include "InteractionModel.H"
#include "DiseaseParm.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief Class describing agent interactions and treatment at hospital */
template <typename AC, typename ACT, typename ACTD, typename A>
class HospitalModel : public InteractionModel<AC,ACT,ACTD,A>
{
    public:

        /*! \brief null constructor */
        HospitalModel() { }

        /*! \brief default destructor */
        virtual ~HospitalModel() = default;

        /*! \brief Simulate agent interaction at hospital */
        virtual void interactAgents( AC&, MultiFab& ) override
        {
            // not yet implemented
        }

        using MFPtr = std::unique_ptr<MultiFab>;
        using MFPtrVec = std::vector<MFPtr>;

        /*! \brief Simulate treatment of agents in a hospital */
        virtual void treatAgents( AC&, MFPtrVec& );

    protected:

    private:
};

/*! Simulate the treatment of agents and interactions between agents at a hospital
   and compute the infection probability for each agent:

    + For each agent *i*:
      + If the agent is not hospitalized, do nothing.
      + Else: implement the treatment
              discharge once treatment is complete
*/
template <typename AC, typename ACT, typename ACTD, typename A>
void HospitalModel<AC,ACT,ACTD,A>::treatAgents(AC& a_agents, /*!< Agent container */
                                               MFPtrVec& a_dstats /*!< disease stats */ )
{
    BL_PROFILE("HospitalModel::interactAgents");
    int n_disease = a_agents.numDiseases();

    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const auto dx = a_agents.Geom(lev).CellSizeArray();
        auto& plev  = a_agents.GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& soa   = ptile.GetStructOfArrays();
            auto& aos   = ptile.GetArrayOfStructs();
            A* pstruct = &(aos[0]);
            const auto np = ptile.numParticles();

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;

            GpuArray<int*,ExaEpi::max_num_diseases> status_ptrs, symptomatic_ptrs;
            GpuArray<ParticleReal*,ExaEpi::max_num_diseases> counter_ptrs, timer_ptrs, incub_per_ptrs;
            for (int d = 0; d < n_disease; d++) {
                status_ptrs[d] = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();
                symptomatic_ptrs[d] = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::symptomatic).data();
                counter_ptrs[d] = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
                timer_ptrs[d] = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::treatment_timer).data();
                incub_per_ptrs[d] = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::incubation_period).data();
            }

            Gpu::DeviceVector<int> is_alive;
            is_alive.resize(np);
            auto is_alive_ptr = is_alive.data();

            ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                // if status_ptr for any one disease is dead, they should all be dead
                if (status_ptrs[0][i] == Status::dead) {
                    is_alive_ptr[i] = 0;
                    for (int d = 1; d < n_disease; d++) {
                        AMREX_ALWAYS_ASSERT(status_ptrs[d][i] == Status::dead);
                    }
                } else {
                    is_alive_ptr[i] = 1;
                    for (int d = 1; d < n_disease; d++) {
                        AMREX_ALWAYS_ASSERT(status_ptrs[d][i] != Status::dead);
                    }
                }
            });
            Gpu::synchronize();

            for (int d = 0; d < n_disease; d++) {

                auto* disease_parm_h = a_agents.getDiseaseParameters_h(d);
                auto mean_immune_time = disease_parm_h->mean_immune_time;
                auto immune_time_spread = disease_parm_h->immune_time_spread;
                auto* disease_parm_d = a_agents.getDiseaseParameters_d(d);

                auto ds_arr = (*a_dstats[d])[mfi].array();

                ParallelForRNG( np,
                                [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine)
                                noexcept
                {
                    if ( !isHospitalized(i, ptd) )  {
                        // agent is not in hospital
                        return;
                    }
                    if (counter_ptrs[d][i] == Math::ceil(incub_per_ptrs[d][i])) {
                        // agent just started treatment
                        return;
                    }
                    if ( timer_ptrs[d][i] == 0) {
                        // agent has recovered/died from disease d
                        return;
                    }
                    if ( is_alive_ptr[i] == 0) {
                        // agent is dead
                        return;
                    }

                    AMREX_ALWAYS_ASSERT(status_ptrs[d][i] == Status::infected);
                    timer_ptrs[d][i] -= 1.0_rt;
                    if (timer_ptrs[d][i] == 0) {
                        disease_parm_d->check_survival( is_alive_ptr[i],
                                                        age_group_ptr[i],
                                                        2.0_rt,
                                                        engine );
                        if (is_alive_ptr[i] == 0) {
                            Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                             home_j_ptr[i],
                                                             0,
                                                             DiseaseStats::death),
                                                   1.0_rt );
                            status_ptrs[d][i] = Status::dead;
                        } else {
                            // If alive, hospitalized patient recovers
                            status_ptrs[d][i] = Status::immune;
                            counter_ptrs[d][i] = (mean_immune_time - immune_time_spread)
                                             + 2.0_rt*immune_time_spread*Random(engine);
                            symptomatic_ptrs[d][i] = SymptomStatus::presymptomatic;
                            withdrawn_ptr[i] = 0;
                            timer_ptrs[d][i] = 0.0_rt;
                        }
                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                         home_j_ptr[i],
                                                         0,
                                                         DiseaseStats::hospitalization ),
                                               -1.0_rt );
                    }
                    if (timer_ptrs[d][i] == 10) {
                        disease_parm_d->check_survival( is_alive_ptr[i],
                                                        age_group_ptr[i],
                                                        1.0_rt,
                                                        engine );
                        if (is_alive_ptr[i] == 0) {
                            Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                             home_j_ptr[i],
                                                             0,
                                                             DiseaseStats::death),
                                                   1.0_rt );
                            status_ptrs[d][i] = Status::dead;
                        } else {
                            // If alive, hospitalized patient recovers
                            status_ptrs[d][i] = Status::immune;
                            counter_ptrs[d][i] = (mean_immune_time - immune_time_spread)
                                             + 2.0_rt*immune_time_spread*Random(engine);
                            symptomatic_ptrs[d][i] = SymptomStatus::presymptomatic;
                            withdrawn_ptr[i] = 0;
                            timer_ptrs[d][i] = 0.0_rt;
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
                    }
                    if (timer_ptrs[d][i] == 20) {
                        disease_parm_d->check_survival( is_alive_ptr[i],
                                                        age_group_ptr[i],
                                                        0.0_rt,
                                                        engine );
                        if (is_alive_ptr[i] == 0) {
                            Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                             home_j_ptr[i],
                                                             0,
                                                             DiseaseStats::death),
                                                   1.0_rt );
                            status_ptrs[d][i] = Status::dead;
                        } else {
                            // If alive, hospitalized patient recovers
                            status_ptrs[d][i] = Status::immune;
                            counter_ptrs[d][i] = (mean_immune_time - immune_time_spread)
                                             + 2.0_rt*immune_time_spread*Random(engine);
                            symptomatic_ptrs[d][i] = SymptomStatus::presymptomatic;
                            withdrawn_ptr[i] = 0;
                            timer_ptrs[d][i] = 0.0_rt;
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
                    }
                });
                Gpu::synchronize();
            }

            ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                if ( !isHospitalized(i, ptd) )  { return; }

                if (is_alive_ptr[i] == 0) {

                    // agent has died
                    for (int d = 0; d < n_disease; d++) {
                        status_ptrs[d][i] = Status::dead;
                    }
                    hosp_i_ptr[i] = -1;
                    hosp_j_ptr[i] = -1;
                    withdrawn_ptr[i] = 0;

                } else {

                    // check if agent can be discharged from hospital
                    ParticleReal sum_timers = 0;
                    for (int d = 0; d < n_disease; d++) {
                        sum_timers += timer_ptrs[d][i];
                    }
                    if (sum_timers == 0) {
                        // discharge patient
                        hosp_i_ptr[i] = -1;
                        hosp_j_ptr[i] = -1;
                        withdrawn_ptr[i] = 0;
                        A& p = pstruct[i];
                        p.pos(0) = (home_i_ptr[i] + 0.5_prt)*dx[0];
                        p.pos(1) = (home_j_ptr[i] + 0.5_prt)*dx[1];
                    }
                }

            });
            Gpu::synchronize();

        }
    }
}

#endif
