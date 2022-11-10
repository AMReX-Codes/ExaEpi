#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "AgentContainer.H"

using namespace amrex;

struct TestParams
{
    IntVect size;
    int max_grid_size;
    int nsteps;
    int plot_int;
};

void get_test_params(TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("nsteps", params.nsteps);

    params.plot_int = -1;
    pp.query("plot_int", params.plot_int);
}

void runAgent();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    runAgent();

    amrex::Finalize();
}

void writePlotFile (AgentContainer& pc, int step) {
    amrex::Print() << "Writing plotfile \n";

    MultiFab particle_count(pc.ParticleBoxArray(0),
                      pc.ParticleDistributionMap(0), 5, 0);
    particle_count.setVal(0.0);
    pc.generateCellData(particle_count);
    WriteSingleLevelPlotfile(amrex::Concatenate("plt", step, 5), particle_count,
                             {"total", "never_infected", "infected", "immune", "previously_infected"},
                             pc.ParticleGeom(0), 0.0, 0);

    // uncomment this to write all the particles
    //pc.WritePlotFile(amrex::Concatenate("plt", step, 5), "agents");
}

void runAgent ()
{
    BL_PROFILE("runAgent");
    TestParams params;
    get_test_params(params, "agent");

    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++) {
        is_per[i] = true;
    }

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        real_box.setLo(n, 0.0);
        real_box.setHi(n, 3000.0);
    }

    IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(params.size[0]-1,params.size[1]-1,params.size[2]-1));
    const Box base_domain(domain_lo, domain_hi);

    Geometry geom;
    geom.define(base_domain, &real_box, CoordSys::cartesian, is_per);

    BoxArray ba;
    DistributionMapping dm;
    IntVect lo = IntVect(AMREX_D_DECL(0, 0, 0));
    ba.define(Box(lo, lo+params.size-1));
    ba.maxSize(params.max_grid_size);
    dm.define(ba);

    AgentContainer pc(geom, dm, ba);

    {
        BL_PROFILE_REGION("Initialization");
        pc.initAgents();
    }

    {
        BL_PROFILE_REGION("Evolution");
        for (int i = 0; i < params.nsteps; ++i)
        {
            amrex::Print() << "Taking step " << i << "\n";

            if (i % 168 == 0) { writePlotFile(pc, i); }  // every week

            pc.updateStatus();
            pc.interactAgents();

            pc.moveAgents();
            if (i % 24 == 0) { pc.moveRandomTravel(); }  // once a day

            pc.Redistribute();

            pc.printTotals();
        }
    }

    if (params.nsteps % 168 == 0) { writePlotFile(pc, params.nsteps); }
}
