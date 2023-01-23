#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "AgentContainer.H"
#include "DemographicData.H"

using namespace amrex;

/**
  * \brief enum for the different initial condition options.
  *        demo is for an initial power law
  *        census reads in census data.
  *
  *        default is demo.
  */
struct ICType {
    enum {
        Demo = 0,
        Census = 1
    };
};

struct TestParams
{
    IntVect size;
    int max_grid_size;
    int nsteps;
    int plot_int;
    short ic_type;
    std::string census_filename;
};

void get_test_params(TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    pp.get("size", params.size);
    pp.get("max_grid_size", params.max_grid_size);
    pp.get("nsteps", params.nsteps);

    params.plot_int = -1;
    pp.query("plot_int", params.plot_int);

    std::string ic_type = "demo";
    pp.query( "ic_type", ic_type );
    if (ic_type == "demo") {
        params.ic_type = ICType::Demo;
    } else if (ic_type == "census") {
        params.ic_type = ICType::Census;
        pp.get("census_filename", params.census_filename);
    } else {
        amrex::Abort("ic type not recognized");
    }
}

void runAgent();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    runAgent();

    amrex::Finalize();
}

/* Determine number of cells in each direction required */
Geometry get_geometry (const DemographicData& demo,
                       const TestParams& params) {
    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++) {
        is_per[i] = true;
    }

    RealBox real_box;
    Box base_domain;
    Geometry geom;

    if (params.ic_type == ICType::Demo) {
        IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
        IntVect domain_hi(AMREX_D_DECL(params.size[0]-1,params.size[1]-1,params.size[2]-1));
        base_domain = Box(domain_lo, domain_hi);

        for (int n = 0; n < BL_SPACEDIM; n++)
        {
            real_box.setLo(n, 0.0);
            real_box.setHi(n, 3000.0);
        }

    } else if (params.ic_type == ICType::Census) {
        IntVect iv;
        iv[0] = iv[1] = (int) std::floor(std::sqrt((double) demo.Ncommunity));
        while (iv[0]*iv[1] <= demo.Ncommunity) {
            ++iv[0];
        }
        base_domain = Box(IntVect(AMREX_D_DECL(0, 0, 0)), iv-1);

        for (int n = 0; n < BL_SPACEDIM; n++)
        {
            real_box.setLo(n, 0.0);
            real_box.setHi(n, 1.0);
        }
    }

    geom.define(base_domain, &real_box, CoordSys::cartesian, is_per);
    return geom;
}

void writePlotFile (AgentContainer& pc, iMultiFab& /*num_residents*/, iMultiFab& unit_mf,
                    iMultiFab& FIPS_mf, iMultiFab& comm_mf, int step) {
    amrex::Print() << "Writing plotfile \n";

    MultiFab output_mf(pc.ParticleBoxArray(0),
                       pc.ParticleDistributionMap(0), 9, 0);
    output_mf.setVal(0.0);
    pc.generateCellData(output_mf);

    amrex::Copy(output_mf, unit_mf, 0, 5, 1, 0);
    amrex::Copy(output_mf, FIPS_mf, 0, 6, 2, 0);
    amrex::Copy(output_mf, comm_mf, 0, 8, 1, 0);

    WriteSingleLevelPlotfile(amrex::Concatenate("plt", step, 5), output_mf,
                             {"total", "never_infected", "infected", "immune", "previously_infected", "unit", "FIPS", "Tract", "comm"},
                             pc.ParticleGeom(0), 0.0, 0);

    // uncomment this to write all the particles
    pc.WritePlotFile(amrex::Concatenate("plt", step, 5), "agents");
}

void runAgent ()
{
    BL_PROFILE("runAgent");
    TestParams params;
    get_test_params(params, "agent");

    DemographicData demo;
    if (params.ic_type == ICType::Census) { demo.InitFromFile(params.census_filename); }
    Geometry geom = get_geometry(demo, params);

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
        }
    }

    {
        BL_PROFILE_REGION("Evolution");
        for (int i = 0; i < params.nsteps; ++i)
        {
            amrex::Print() << "Taking step " << i << "\n";

            if (i % 168 == 0) {
                writePlotFile(pc, num_residents, unit_mf, FIPS_mf, comm_mf, i);
            }  // every week

            pc.updateStatus();
            pc.interactAgents();

            pc.moveAgents();
            if (i % 24 == 0) { pc.moveRandomTravel(); }  // once a day

            pc.Redistribute();

            pc.printTotals();
        }
    }

    if (params.nsteps % 168 == 0) {
        writePlotFile(pc, num_residents, unit_mf, FIPS_mf, comm_mf, params.nsteps);
    }
}
