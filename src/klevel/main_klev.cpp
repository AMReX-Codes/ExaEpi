#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include "AgentContainerWrapper.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "Initialization.H"
#include "IO.H"
#include "Utils.H"

#include <AMReX_Amr.H>
#include <AMReX_Box.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_LevelBld.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

using namespace amrex;
using namespace ExaEpi;

void runAgent();


void override_parameters ()
{
    {
        ParmParse pp("amrex");
        if (!pp.contains("abort_on_out_of_gpu_memory")) {
            // Abort if we run out of GPU memory.
            pp.add("abort_on_out_of_gpu_memory", true);
        }
    }
    {
        ParmParse pp("amr");
        // Always check for whether to dump a plotfile or checkpoint.
        if (!pp.contains("message_int")) {
            pp.add("message_int", 1);
        }
    }
    {
        TestParams params;
        ExaEpi::Utils::get_test_params(params, "agent");
        DemographicData demo;
        if (params.ic_type == ICType::Census) { demo.InitFromFile(params.census_filename); }
        CaseData cases;
        if (params.ic_type == ICType::Census) { cases.InitFromFile(params.case_filename); }
        {
            ParmParse pp("agent");
   	    int max_grid_size=-1;
	    pp.get("max_grid_size", max_grid_size);
	    int ncells= std::floor(std::sqrt((double) demo.Ncommunity));
	    ncells*= max_grid_size; 
            ParmParse ppa("amr");
            ppa.addarr("n_cell", std::vector{ncells, ncells});
	    ppa.add("max_grid_size", max_grid_size);
        }
    }
}


class AgentBld
    :
    public amrex::LevelBld
{
    void variableSetUp (){amrex::Print()<<"seting up AgentBld\n";};
    void variableCleanUp () {amrex::Print()<<"Cleaning up AgentBld\n";};
    AmrLevel *operator() () {return new AgentLevel;};
    AmrLevel *operator() (Amr&            papa,
                          int             lev,
                          const Geometry& level_geom,
                          const BoxArray& ba,
                          const DistributionMapping& dm,
                          Real            time){
	    		  return new AgentLevel(papa, lev, level_geom, ba, dm, time);
    };
};


AgentBld Agent_bld;

int main (int argc, char* argv[])
{
    Real strt_time;
    Real stop_time;
    int  max_step=1000;
    strt_time =  0.0;
    stop_time = -1.0;    
    
    //amrex::Initialize(argc,argv);
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, override_parameters);    
    Amr* amrptr = new Amr(&Agent_bld);
    amrptr->init(strt_time,stop_time);    
    amrex::ParmParse pp("agent");
    pp.get("max_step",max_step);
    pp.get("start_time",strt_time);
    pp.get("stop_time",stop_time);
    while ( amrptr->okToContinue()                            &&
           (amrptr->levelSteps(0) < max_step || max_step < 0) &&
           (amrptr->cumTime() < stop_time || stop_time < 0.0) )

    {
        // Do a coarse timestep (state/national scale on the base level and 
        amrptr->coarseTimeStep(stop_time);
    }
    amrex::Finalize();
}
