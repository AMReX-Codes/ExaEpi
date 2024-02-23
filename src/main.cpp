/*! @file main.cpp
    \brief **Main**: Contains main() and runAgent()
*/

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

/*! \brief Set ExaEpi-specific defaults for memory-management */
void override_amrex_defaults ()
{
    amrex::ParmParse pp("amrex");

    // ExaEpi currently assumes we have mananaged memory in the Arena
    bool the_arena_is_managed = true;
    pp.queryAdd("the_arena_is_managed", the_arena_is_managed);
}

/*! \brief Main function: initializes AMReX, calls runAgent(), finalizes AMReX */
int main (int argc, /*!< Number of command line arguments */
          char* argv[] /*!< Command line arguments */)
{
    amrex::Initialize(argc,argv,true,MPI_COMM_WORLD,override_amrex_defaults);

    runAgent();

    amrex::Finalize();
}

/*! \brief Run agent-based simulation:

    \b Initialization
    + Read test parameters (#ExaEpi::TestParams) from command line input file
    + If initialization type (#ExaEpi::TestParams::ic_type) is ExaEpi::ICType::Census,
      + Read #DemographicData from #ExaEpi::TestParams::census_filename
        (see DemographicData::InitFromFile)
      + Read #CaseData from #ExaEpi::TestParams::case_filename
        (see CaseData::InitFromFile)
    + Get computational domain from ExaEpi::Utils::get_geometry. Each grid cell corresponds to
      a community.
    + Create box arrays and distribution mapping based on #ExaEpi::TestParams::max_grid_size.
    + Initialize the following MultiFabs:
      + Number of residents: 6 components - number of residents in age groups under-5, 5-17,
        18-29, 30-64, 65+, total.
      + Unit number of the community at each grid cell (1 component).
      + FIPS code of the community at each grid cell (2 components - FIPS code, census tract ID).
      + Community number of the community at each grid cell.
      + Disease statistics with 4 components (hospitalization, ICU, ventilator, deaths)
      + Masking behavior
    + Initialize agents (AgentContainer::initAgentsDemo or AgentContainer::initAgentsCensus).
      If ExaEpi::TestParams::ic_type is ExaEpi::ICType::Census, then
      + Read worker flow (ExaEpi::Initialization::read_workerflow)
      + Initialize cases (ExaEpi::Initialization::setInitialCases)


    \b Evolution
    At each step from 0 to #ExaEpi::TestParams::nsteps-1:
    + IO:
      + if the current step number is a multiple of #ExaEpi::TestParams::plot_int, then write
        out plot file - see ExaEpi::IO::writePlotFile()
      + if current step number is a multiple of #ExaEpi::TestParams::aggregated_diag_int, then write
        out aggregated diagnostic data - see ExaEpi::IO::writeFIPSData().
    + Agents behavior:
      + Update agent #Status based on their age, number of days since infection, hospitalization,
        etc. - see AgentContainer::updateStatus().
      + Move agents to work - see AgentContainer::moveAgentsToWork().
      + Let agents interact at work - see AgentContainer::interactAgentsHomeWork().
      + Move agents to home - see AgentContainer::moveAgentsToHome().
      + Let agents interact at home - see AgentContainer::interactAgentsHomeWork().
      + Infect agents based on their movements during the day - see AgentContainer::infectAgents().
    + Get disease statistics counts - see AgentContainer::printTotals() - and update the
      peak number of infections and cumulative deaths.

    \b Finalize
    + Report peak infections, day of peak infections, and cumulative deaths.
    + Write out final plot file - see ExaEpi::IO::writePlotFile()
    + Write out final aggregated diagnostic data - see ExaEpi::IO::writeFIPSData().
*/
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

    amrex::Print() << "Base domain is: " << geom.Domain() << "\n";
    amrex::Print() << "Max grid size is: " << params.max_grid_size << "\n";
    amrex::Print() << "Number of boxes is: " << ba.size() << " over " << ParallelDescriptor::NProcs() << " ranks. \n";

    iMultiFab num_residents(ba, dm, 6, 0);
    iMultiFab unit_mf(ba, dm, 1, 0);
    iMultiFab FIPS_mf(ba, dm, 2, 0);
    iMultiFab comm_mf(ba, dm, 1, 0);

    MultiFab disease_stats(ba, dm, 4, 0);
    MultiFab mask_behavior(ba, dm, 1, 0);
    mask_behavior.setVal(1);

    AgentContainer pc(geom, dm, ba);

    {
        BL_PROFILE_REGION("Initialization");
        if (params.ic_type == ICType::Demo) {
            pc.initAgentsDemo(num_residents, unit_mf, FIPS_mf, comm_mf, demo);
        } else if (params.ic_type == ICType::Census) {
            pc.initAgentsCensus(num_residents, unit_mf, FIPS_mf, comm_mf, demo);
            ExaEpi::Initialization::read_workerflow(demo, params, unit_mf, comm_mf, pc);
            ExaEpi::Initialization::setInitialCases(pc, unit_mf, FIPS_mf, comm_mf, cases, demo);
        }
    }

    int  step_of_peak = 0;
    Long num_infected_peak = 0;
    Long cumulative_deaths = 0;
    {
        auto counts = pc.printTotals();
        if (counts[1] > num_infected_peak) {
            num_infected_peak = counts[1];
            step_of_peak = 0;
        }
        cumulative_deaths = counts[4];
    }

    amrex::Real cur_time = 0;
    {
        BL_PROFILE_REGION("Evolution");
        for (int i = 0; i < params.nsteps; ++i)
        {
            amrex::Print() << "Simulating day " << i << "\n";

            if ((params.plot_int > 0) && (i % params.plot_int == 0)) {
                ExaEpi::IO::writePlotFile(pc, num_residents, unit_mf, FIPS_mf, comm_mf, cur_time, i);
            }

            if ((params.aggregated_diag_int > 0) && (i % params.aggregated_diag_int == 0)) {
                ExaEpi::IO::writeFIPSData(pc, unit_mf, FIPS_mf, comm_mf, demo, params.aggregated_diag_prefix, i);
            }

            // Update agents' disease status
            pc.updateStatus(disease_stats);

            // Typical day
            pc.morningCommute(mask_behavior);
            pc.interactDay(mask_behavior);
            pc.eveningCommute(mask_behavior);
            pc.interactEvening(mask_behavior);
            pc.interactNight(mask_behavior);

            // Infect agents based on their interactions
            pc.infectAgents();

            //            if ((params.random_travel_int > 0) && (i % params.random_travel_int == 0)) {
            //                pc.moveRandomTravel();
            //            }
            //            pc.Redistribute();

            auto counts = pc.printTotals();
            if (counts[1] > num_infected_peak) {
                num_infected_peak = counts[1];
                step_of_peak = i;
            }
            cumulative_deaths = counts[4];

            amrex::Print() << "    Total never: "       << counts[0] << "\n";
            amrex::Print() << "    Total infected: "    << counts[1] << "\n";
            amrex::Print() << "    Total immune: "      << counts[2] << "\n";
         // amrex::Print() << "    Total susceptible: " << counts[3] << "\n";
            amrex::Print() << "    Total deaths: "      << counts[4] << "\n";
            amrex::Print() << "\n";

            cur_time += 1.0; // time step is one day
        }
    }

    amrex::Print() << "\n \n";
    amrex::Print() << "Peak number of infected: " << num_infected_peak << "\n";
    amrex::Print() << "Day of peak: " << step_of_peak << "\n";
    amrex::Print() << "Cumulative deaths: " << cumulative_deaths << "\n";
    amrex::Print() << "\n \n";

    if (params.plot_int > 0) {
        ExaEpi::IO::writePlotFile(pc, num_residents, unit_mf, FIPS_mf, comm_mf, cur_time, params.nsteps);
    }

    if ((params.aggregated_diag_int > 0) && (params.nsteps % params.aggregated_diag_int == 0)) {
        ExaEpi::IO::writeFIPSData(pc, unit_mf, FIPS_mf, comm_mf, demo, params.aggregated_diag_prefix, params.nsteps);
    }
}
