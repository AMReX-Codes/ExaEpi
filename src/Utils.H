#ifndef UTILS_H_
#define UTILS_H_

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>

#include "DemographicData.H"

#include <string>

namespace ExaEpi
{

struct TestParams
{
    amrex::IntVect size;
    int max_grid_size;
    int nsteps;
    int plot_int;
    short ic_type;
    std::string census_filename;
    std::string workerflow_filename;
};

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

namespace Utils
{

    void get_test_params (ExaEpi::TestParams& params, const std::string& prefix);

    /* Determine number of cells in each direction required */
    amrex::Geometry get_geometry (const DemographicData& demo,
                                  const ExaEpi::TestParams& params);

}
}

#endif
