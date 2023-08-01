#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#include "AgentContainer.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "Initialization.H"
#include "IO.H"
#include "Utils.H"

using namespace amrex;
using namespace ExaEpi;

void runAgent();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    runAgent();

    amrex::Finalize();
}

void runAgent ()
{
    BL_PROFILE("runAgent");
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

    AgentContainer pc(geom, dm, ba);

    {
        BL_PROFILE_REGION("Initialization");
        if (params.ic_type == ICType::Demo) {
            pc.initAgentsDemo(num_residents, unit_mf, FIPS_mf, comm_mf, demo);
        } else if (params.ic_type == ICType::Census) {
            pc.initAgentsCensus(num_residents, unit_mf, FIPS_mf, comm_mf, demo);
            ExaEpi::Initialization::read_workerflow(demo, params, unit_mf, comm_mf, pc);
            //ExaEpi::Initialization::setInitialCases(pc, unit_mf, FIPS_mf, comm_mf, cases, demo);
        }
    }

    {
        BL_PROFILE_REGION("Evolution");
        for (int i = 0; i < params.nsteps; ++i)
        {
            amrex::Print() << "Taking step " << i << "\n";

            if ((params.plot_int > 0) && (i % params.plot_int == 0)) {
                ExaEpi::IO::writePlotFile(pc, num_residents, unit_mf, FIPS_mf, comm_mf, i);
            }

            if ((params.aggregated_diag_int > 0) && (i % params.aggregated_diag_int == 0)) {
                ExaEpi::IO::writeFIPSData(pc, unit_mf, FIPS_mf, comm_mf, demo, params.aggregated_diag_prefix, i);
            }

            pc.updateStatus();
            pc.interactAgents();

            pc.moveAgentsRandomWalk();
            if ((params.random_travel_int > 0) && (i % params.random_travel_int == 0)) {
                pc.moveRandomTravel();
            }

            pc.Redistribute();

            pc.printTotals();
        }
    }

    if ((params.plot_int > 0) && (params.nsteps % params.plot_int == 0)) {
        ExaEpi::IO::writePlotFile(pc, num_residents, unit_mf, FIPS_mf, comm_mf, params.nsteps);
    }

    if ((params.aggregated_diag_int > 0) && (params.nsteps % params.aggregated_diag_int == 0)) {
        ExaEpi::IO::writeFIPSData(pc, unit_mf, FIPS_mf, comm_mf, demo, params.aggregated_diag_prefix, params.nsteps);
    }
}
