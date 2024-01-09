#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_CoordSys.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>


#include "DemographicData.H"
#include "Utils.H"

#include <cmath>
#include <string>

using namespace amrex;
using namespace ExaEpi;

void ExaEpi::Utils::get_test_params (TestParams& params, const std::string& prefix)
{
    ParmParse pp(prefix);
    params.size = {1, 1};
    pp.query("size", params.size);

    params.max_grid_size = 16;
    pp.query("max_grid_size", params.max_grid_size);

    pp.get("nsteps", params.nsteps);

    params.plot_int = -1;
    pp.query("plot_int", params.plot_int);

    params.random_travel_int = -1;
    pp.query("random_travel_int", params.random_travel_int);

    std::string ic_type = "demo";
    pp.query( "ic_type", ic_type );
    if (ic_type == "demo") {
        params.ic_type = ICType::Demo;
    } else if (ic_type == "census") {
        params.ic_type = ICType::Census;
        pp.get("census_filename", params.census_filename);
        pp.get("workerflow_filename", params.workerflow_filename);
        pp.get("case_filename", params.case_filename);
    } else {
        amrex::Abort("ic type not recognized");
    }

    params.aggregated_diag_int = -1;
    pp.query("aggregated_diag_int", params.aggregated_diag_int);
    if (params.aggregated_diag_int >= 0) {
        pp.get("aggregated_diag_prefix", params.aggregated_diag_prefix);
    }

    Long seed = 0;
    bool reset_seed = pp.query("seed", seed);
    if (reset_seed) {
        ULong gpu_seed = (ULong) seed;
        ULong cpu_seed = (ULong) seed;
        amrex::ResetRandomSeed(cpu_seed, gpu_seed);
    }
}

/* Determine number of cells in each direction required */
Geometry ExaEpi::Utils::get_geometry (const DemographicData& demo,
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
