/*! @file IO.cpp
    \brief Contains IO functions in #ExaEpi::IO namespace
*/

#include <AMReX_GpuContainers.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include "IO.H"

#include <vector>

using namespace amrex;

namespace ExaEpi
{
namespace IO
{

/*! \brief Write plotfile of computational domain with disease spread and census data at a given step.

    Writes the current disease spread information and census data (unit, FIPS code, census tract ID,
    and community number) to a plotfile:
    + Create an output MultiFab (with the same domain and distribution map as the particle container)
      with 9 components:
      + component 0: total
      + component 1: never infected (#Status::never)
      + component 2: infected (#Status::infected)
      + component 3: immune (#Status::immune)
      + component 4: previously infected (#Status::susceptible QDG??)
      + component 5: unit number
      + component 6: FIPS ID
      + component 7: census tract number
      + component 8: community number
    + Get disease spread data (first 5 components) from AgentContainer::generateCellData().
    + Copy unit number, FIPS code, census tract ID, and community number from the input MultiFabs to
      the remaining components.
    + Write the output MultiFab to file.
    + Write agents to file - see AgentContainer::WritePlotFile().
*/
void writePlotFile (const AgentContainer& pc,   /*!< Agent (particle) container */
                    const iMultiFab& /*num_residents*/,
                    const iMultiFab& unit_mf,   /*!< MultiFab with unit number of each community */
                    const iMultiFab& FIPS_mf,   /*!< MultiFab with FIPS code and census tract ID */
                    const iMultiFab& comm_mf,   /*!< MultiFab of community number */
                    const int step              /*!< Current step */) {
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

/*! \brief Writes diagnostic data by FIPS code

    Writes a file with the total number of infected agents for each unit;
    it writes out the number of infected agents in the same order as the units in the
    census data file.
    + Creates a output vector of size #DemographicData::Nunit (total number of units).
    + Gets the disease status in agents from AgentContainer::generateCellData().
    + On each processor, sets the unit-th element of the output vector to the number of
      infected agents in the communities on this processor belonging to that unit.
    + Sum across all processors and write to file.
*/
void writeFIPSData (const AgentContainer& agents, /*!< Agents (particle) container */
                    const iMultiFab& unit_mf,     /*!< MultiFab with unit number of each community */
                    const iMultiFab& /*FIPS_mf*/,
                    const iMultiFab& /*comm_mf*/,
                    const DemographicData& demo,  /*!< Demographic data */
                    const std::string& prefix,    /*!< Filename prefix */
                    const int step                /*!< Current step */) {
    amrex::Print() << "Generating diagnostic data by FIPS code \n";

    std::vector<amrex::Real> data(demo.Nunit, 0.0);
    amrex::Gpu::DeviceVector<amrex::Real> d_data(data.size(), 0.0);
    amrex::Real* const AMREX_RESTRICT data_ptr = d_data.dataPtr();

    int const nlevs = std::max(0, agents.finestLevel()+1);
    for (int lev = 0; lev < nlevs; ++lev) {
        MultiFab mf(agents.ParticleBoxArray(lev),
                    agents.ParticleDistributionMap(lev), 9, 0);
        mf.setVal(0.0);
        agents.generateCellData(mf);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        {
            for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                auto unit_arr = unit_mf[mfi].array();
                auto cell_data_arr = mf[mfi].array();

                auto bx = mfi.tilebox();
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        int unit = unit_arr(i, j, k);  // which FIPS
                        int num_infected = int(cell_data_arr(i, j, k, 2));
                        amrex::Gpu::Atomic::AddNoRet(&data_ptr[unit], (amrex::Real) num_infected);
                    });
            }
        }
    }

    // blocking copy from device to host
    amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                     d_data.begin(), d_data.end(), data.begin());

    // reduced sum over mpi ranks
    ParallelDescriptor::ReduceRealSum
        (data.data(), data.size(), ParallelDescriptor::IOProcessorNumber());

    if (ParallelDescriptor::IOProcessor())
    {
        std::string fn = amrex::Concatenate(prefix, step, 5);
        std::ofstream ofs{fn, std::ofstream::out | std::ofstream::app};

        // set precision
        ofs << std::fixed << std::setprecision(14) << std::scientific;

        // loop over data size and write
        for (const auto& item : data) {
            ofs << " " << item;
        }

        ofs << std::endl;
        ofs.close();
    }
}

}
}
