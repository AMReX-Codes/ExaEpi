#include <AMReX_GpuContainers.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_REAL.H>

#include "IO.H"

#include <vector>

using namespace amrex;

namespace ExaEpi
{
namespace IO
{

void writePlotFile (const AgentContainer& pc, const iMultiFab& /*num_residents*/, const iMultiFab& unit_mf,
                    const iMultiFab& FIPS_mf, const iMultiFab& comm_mf, const int step) {
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

void writeFIPSData (const AgentContainer& agents, const iMultiFab& unit_mf,
                    const iMultiFab& FIPS_mf, const iMultiFab& comm_mf,
                    const DemographicData& demo) {
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
                        int unit = unit_arr(i, j, k);
                        int num_infected = cell_data_arr(i, j, k, 2);
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
        // open file
        std::ofstream ofs{"test.txt",
            std::ofstream::out | std::ofstream::app};

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
