/*! @file Utils.cpp
    \brief Contains function implementations for the #ExaEpi::Utils namespace
*/

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

/*! \brief Read in test parameters in #ExaEpi::TestParams from input file */
void ExaEpi::Utils::get_test_params (   TestParams& params,         /*!< Test parameters */
                                        const std::string& prefix   /*!< ParmParse prefix */ )
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

    params.num_diseases = 1;
    pp.query("number_of_diseases", params.num_diseases);

    params.disease_names.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        params.disease_names[d] = amrex::Concatenate("default", d, 2);
    }
    pp.queryarr("disease_names", params.disease_names,0,params.num_diseases);

    params.initial_case_type.resize(params.num_diseases);
    params.num_initial_cases.resize(params.num_diseases);
    params.case_filename.resize(params.num_diseases);

    std::string ic_type = "demo";
    pp.query( "ic_type", ic_type );
    if (ic_type == "demo") {
        params.ic_type = ICType::Demo;
    } else if (ic_type == "census") {
        params.ic_type = ICType::Census;
        pp.get("census_filename", params.census_filename);
        pp.get("workerflow_filename", params.workerflow_filename);
        pp.getarr("initial_case_type", params.initial_case_type,0,params.num_diseases);
        if (params.num_diseases == 1) {
            if (params.initial_case_type[0] == "file") {
                if (pp.contains("case_filename")) {
                    pp.get("case_filename", params.case_filename[0]);
                } else {
                    std::string key = "case_filename_" + params.disease_names[0];
                    pp.get(key.c_str(), params.case_filename[0]);
                }
            } else if (params.initial_case_type[0] == "random") {
                if (pp.contains("num_initial_cases")) {
                    pp.get("num_initial_cases", params.num_initial_cases[0]);
                } else {
                    std::string key = "num_initial_cases_" + params.disease_names[0];
                    pp.get(key.c_str(), params.num_initial_cases[0]);
                }
            } else {
                amrex::Abort("initial case type not recognized");
            }
        } else {
            for (int d = 0; d < params.num_diseases; d++) {
                if (params.initial_case_type[d] == "file") {
                    std::string key = "case_filename_" + params.disease_names[d];
                    pp.get(key.c_str(), params.case_filename[d]);
                } else if (params.initial_case_type[d] == "random") {
                    std::string key = "num_initial_cases_" + params.disease_names[d];
                    pp.get(key.c_str(), params.num_initial_cases[d]);
                } else {
                    amrex::Abort("initial case type not recognized");
                }
            }
        }
    } else {
        amrex::Abort("ic type not recognized");
    }

    params.aggregated_diag_int = -1;
    pp.query("aggregated_diag_int", params.aggregated_diag_int);
    if (params.aggregated_diag_int >= 0) {
        pp.get("aggregated_diag_prefix", params.aggregated_diag_prefix);
    }

    pp.query("shelter_start",  params.shelter_start);
    pp.query("shelter_length", params.shelter_length);

    Long seed = 0;
    bool reset_seed = pp.query("seed", seed);
    if (reset_seed) {
        ULong gpu_seed = (ULong) seed;
        ULong cpu_seed = (ULong) seed;
        amrex::ResetRandomSeed(cpu_seed, gpu_seed);
    }
}

/*! \brief Set computational domain, i.e., number of cells in each direction, from the
    demographic data (number of communities).
 *
 *  If the initialization type (ExaEpi::TestParams::ic_type) is ExaEpi::ICType::Census, then
 *  + The domain is a 2D square, where the total number of cells is the lowest square of an
 *    integer that is greater than #DemographicData::Ncommunity
 *  + The physical size is 1.0 in each dimension.
 *
 *  A periodic Cartesian grid is defined.
*/
Geometry ExaEpi::Utils::get_geometry (const DemographicData&    demo,   /*!< demographic data */
                                      const TestParams&         params  /*!< test parameters */ ) {
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
