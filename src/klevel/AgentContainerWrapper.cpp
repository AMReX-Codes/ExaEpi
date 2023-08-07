#include "AgentContainerWrapper.H"
#include <AMReX_ParmParse.H>
#include "CaseData.H"
#include "DemographicData.H"
#include "Initialization.H"
#include "IO.H"
#include "Utils.H"

using namespace ExaEpi;
using namespace amrex;
Vector<AMRErrorTag> AgentLevel::error_tags;	

AgentLevel::AgentLevel () {
};

AgentLevel::AgentLevel (amrex::Amr&            papa,
    int             lev,
    const amrex::Geometry& level_geom,
    const amrex::BoxArray& bl,
    const amrex::DistributionMapping& dm,
    amrex::Real time):AmrLevel(papa,lev,level_geom,bl,dm,time)
    {
	buildStateVars();
	if(lev==0){
	    amrex::Print()<<"Setting up ExaEpi variables using Census data\n";
	    variableSetUp();
	}
    };

void AgentLevel::variableSetUp (){
    TestParams params;
    ExaEpi::Utils::get_test_params(params, "agent");
    DemographicData demo;
    if (params.ic_type == ICType::Census) { demo.InitFromFile(params.census_filename); }

    CaseData cases;
    if (params.ic_type == ICType::Census) { cases.InitFromFile(params.case_filename); }
    Geometry geom = ExaEpi::Utils::get_geometry(demo, params);

    BoxArray ba;
    DistributionMapping dm;
    ba.define(geom.Domain());
    ba.maxSize(params.max_grid_size);
    dm.define(ba);

    iMultiFab num_residents(ba, dm, 6, 0);
    iMultiFab unit_mf(ba, dm, 1, 0);
    iMultiFab FIPS_mf(ba, dm, 2, 0);
    iMultiFab comm_mf(ba, dm, 1, 0);

    pc= new AgentContainerWrapper(geom, dm, ba);
    {
        BL_PROFILE_REGION("Initialization");
        if (params.ic_type == ICType::Demo) {
            pc->initAgentsDemo(num_residents, unit_mf, FIPS_mf, comm_mf, demo);
        } else if (params.ic_type == ICType::Census) {
            pc->initAgentsCensus(num_residents, unit_mf, FIPS_mf, comm_mf, demo);
        }
        ExaEpi::Initialization::read_workerflow(demo, params, unit_mf, comm_mf, *pc);
    }
};

void AgentContainerWrapper::copyAgentContainerToAMRGrids(amrex::MultiFab& statusMF, amrex::MultiFab& probMF, amrex::MultiFab& timerMF){
    //we should have a more optimized version of this data copy
    BL_PROFILE("AgentContainer::copyAgentsToAMRGrids");
    const BoxArray& ba = statusMF.boxArray ();
    int numTransferedAgents= ba.numPts();
    Real* status_tmp= new Real[numTransferedAgents];
    Real* prob_tmp= new Real[numTransferedAgents];
    Real* timer_tmp= new Real[numTransferedAgents];
    {
        auto& plev  = GetParticles(0);
	int idx=0;
        for(MFIter mfi = MakeMFIter(0); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.validbox();
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
	    if(idx+np>= numTransferedAgents) break;
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto prob_ptr = soa.GetRealData(RealIdx::prob).data();
            auto timer_ptr = soa.GetRealData(RealIdx::timer).data();
	    Real* s_ptr=&status_tmp[idx];
	    Real* p_ptr=&prob_tmp[idx];
	    Real* t_ptr=&timer_tmp[idx];

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
	        s_ptr[i]  = (Real)status_ptr[i];
	        p_ptr[i]  = prob_ptr[i];
	        t_ptr[i]  = timer_ptr[i];
            });
            idx+=np;
        }
    }

    {
	int idx=0;
        for(MFIter mfi(statusMF); mfi.isValid(); ++mfi)
        {
	    const Box& box = mfi.validbox();
  	    auto status = statusMF.array(mfi);
  	    auto infect = probMF.array(mfi);
  	    auto timer = timerMF.array(mfi);
	    if(idx+ box.length(1)*box.length(0)>= numTransferedAgents) break;
	    Real* s_ptr=&status_tmp[idx];
	    Real* p_ptr=&prob_tmp[idx];
	    Real* t_ptr=&timer_tmp[idx];
            amrex::ParallelFor(box,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                 int i_idx=i-box.smallEnd()[0];
                 int j_idx=j-box.smallEnd()[1];
	         status(i,j,k)= s_ptr[j_idx*box.length(0)+i_idx];
	         infect(i,j,k)= p_ptr[j_idx*box.length(0)+i_idx];
	         timer(i,j,k)= t_ptr[j_idx*box.length(0)+i_idx];
	    });
	    idx+= box.length(1)*box.length(0);
	}
    }
    delete [] status_tmp;
    delete [] prob_tmp;
    delete [] timer_tmp;
}

void AgentContainerWrapper::copyAMRGridsToAgentContainer(amrex::MultiFab& statusMF, amrex::MultiFab& probMF, amrex::MultiFab& timerMF){
    //we should have a more optimized version of this data copy
    BL_PROFILE("AgentContainer::copyAgentsToAMRGrids");
    const BoxArray& ba = statusMF.boxArray ();
    int numTransferedAgents= ba.numPts();
    Real* status_tmp= new Real[numTransferedAgents];
    Real* prob_tmp= new Real[numTransferedAgents];
    Real* timer_tmp= new Real[numTransferedAgents];
    {
        int idx=0;
        for(MFIter mfi(statusMF); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.validbox();
            auto status = statusMF.array(mfi);
            auto infect = probMF.array(mfi);
            auto timer = timerMF.array(mfi);
            if(idx+ box.length(1)*box.length(0)>= numTransferedAgents) break;
            Real* s_ptr=&status_tmp[idx];
            Real* p_ptr=&prob_tmp[idx];
            Real* t_ptr=&timer_tmp[idx];
            amrex::ParallelFor(box,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                 int i_idx=i-box.smallEnd()[0];
                 int j_idx=j-box.smallEnd()[1];
		 s_ptr[j_idx*box.length(0)+i_idx]= status(i,j,k);
		 p_ptr[j_idx*box.length(0)+i_idx]= infect(i,j,k);
		 t_ptr[j_idx*box.length(0)+i_idx]= timer(i,j,k);
            });
            idx+= box.length(1)*box.length(0);
        }
    }
    {
        auto& plev  = GetParticles(0);
        int idx=0;
        for(MFIter mfi = MakeMFIter(0); mfi.isValid(); ++mfi)
        {
            const Box& box = mfi.validbox();
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if(idx+np>= numTransferedAgents) break;
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto prob_ptr = soa.GetRealData(RealIdx::prob).data();
            auto timer_ptr = soa.GetRealData(RealIdx::timer).data();
            Real* s_ptr=&status_tmp[idx];
            Real* p_ptr=&prob_tmp[idx];
            Real* t_ptr=&timer_tmp[idx];

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
		status_ptr[i]= s_ptr[i];
		prob_ptr[i]  = p_ptr[i];
		timer_ptr[i]  = t_ptr[i];
            });
            idx+=np;
        }
    }

    delete [] status_tmp;
    delete [] prob_tmp;
    delete [] timer_tmp;
}

