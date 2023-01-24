#include <AMReX_PlotFileUtil.H>

#include "IO.H"

using namespace amrex;

namespace ExaEpi
{
namespace IO
{

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

}
}
