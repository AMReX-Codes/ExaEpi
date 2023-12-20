/*! @file Utils.H
    \brief Defines the #ExaEpi::Utils namespace
*/

#ifndef UTILS_H_
#define UTILS_H_

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>

#include "DemographicData.H"

#include <string>

namespace ExaEpi
{

/*! \brief Structure with various simulation parameters; these are read in
    for the input file provided in the command line.
*/
struct TestParams
{
    amrex::IntVect size;                /*!< domain size; (QDG: used only for ic_type = demo?)
                                             (see ExaEpi::Utils::get_geometry) */
    int max_grid_size;                  /*!< box size */
    int nsteps;                         /*!< number of simulation steps */
    int plot_int;                       /*!< plot interval (see ExaEpi::IO::writePlotFile) */
    int random_travel_int;              /*!< steps between random travel events
                                             (see AgentContainer::moveRandomTravel) */
    short ic_type;                      /*!< initialization type (see ExaEpi::ICType) */
    std::string census_filename;        /*!< census data filename (DemographicData::InitFromFile) */
    std::string workerflow_filename;    /*!< worker flow filename
                                             (ExaEpi::Initialization::read_workerflow) */
    std::string case_filename;          /*!< initial cases filename (CaseData::InitFromFile) */

    int aggregated_diag_int;            /*!< interval for writing aggregated diagnostic data
                                             (see: ExaEpi::IO::writeFIPSData) */
    std::string aggregated_diag_prefix; /*!< filename prefix for diagnostic data
                                             (see: ExaEpi::IO::writeFIPSData) */
};

/**
  * \brief enum for the different initial condition options.\n
  *        demo is for an initial power law\n
  *        census reads in census data.\n
  *
  *        Default is demo.
  */
struct ICType {
    enum {
        Demo = 0,   /*!< Initial power law (default) */
        Census = 1  /*!< Census data */
    };
};

/*! \brief Namespace with utility functions */
namespace Utils
{

    void get_test_params (ExaEpi::TestParams& params, const std::string& prefix);

    amrex::Geometry get_geometry (const DemographicData& demo,
                                  const ExaEpi::TestParams& params);

}
}

#endif
