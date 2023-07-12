#include "Initialization.H"

#include "DemographicData.H"
#include "Utils.H"
#include "AgentContainer.H"

#include <AMReX_Arena.H>
#include <AMReX_Box.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Random.H>
#include <AMReX_VisMF.H>

#include <fstream>

using namespace amrex;

namespace ExaEpi
{
namespace Initialization
{
    void read_workerflow (const DemographicData& demo,
                          const TestParams& params,
                          const iMultiFab& unit_mf,
                          const iMultiFab& comm_mf,
                          AgentContainer& pc) {

  /* Allocate worker-flow matrix, only from units with nighttime
     communities on this processor (Unit_on_proc[] flag) */
    unsigned int** flow = (unsigned int **) amrex::The_Arena()->alloc(demo.Nunit*sizeof(unsigned int *));
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            flow[i] = (unsigned int *) amrex::The_Arena()->alloc(demo.Nunit*sizeof(unsigned int));
            for (int j = 0; j < demo.Nunit; j++) flow[i][j] = 0;
        }
    }

    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    std::ifstream ifs;
    ifs.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

    ifs.open(params.workerflow_filename.c_str(), std::ios::in|std::ios::binary);
    if (!ifs.good()) {
        amrex::FileOpenFailed(params.workerflow_filename);
    }

    const std::streamoff CURPOS = ifs.tellg();
    ifs.seekg(0,std::ios::end);
    const std::streamoff ENDPOS = ifs.tellg();
    const long num_work = (ENDPOS - CURPOS) / (3*sizeof(unsigned int));

    ifs.seekg(CURPOS, std::ios::beg);

    for (int work = 0; work < num_work; ++work) {
        unsigned int from, to, number;
        ifs.read((char*)&from, sizeof(from));
        ifs.read((char*)&to, sizeof(to));
        ifs.read((char*)&number, sizeof(number));
        int i = demo.myIDtoUnit[from];
        if (demo.Unit_on_proc[i]) {
            int j = demo.myIDtoUnit[to];
            if (demo.Start[j+1] != demo.Start[j]) { // if there are communities in this unit
                flow[i][j] = number;
            }
        }
    }

    /* Convert to cumulative numbers to enable random selection */
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            for (int j = 1; j < demo.Nunit; j++) {
                flow[i][j] += flow[i][j-1];
            }
        }
    }

    /* These numbers were for the true population, and do not include
       the roughly 2% of people who were on vacation or sick during the
       Census 2000 reporting week.  We need to scale the worker flow to
       the model tract residential populations, and might as well add
       the 2% back in while we're at it.... */
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i] && demo.Population[i]) {
            unsigned int number = (unsigned int) rint(((double) demo.Population[i]) / 2000.0);
            double scale = 1.02 * (2000.0 * number) / ((double) demo.Population[i]);
            for (int j = 0; j < demo.Nunit; j++) {
                flow[i][j] = (unsigned int) rint((double) flow[i][j] * scale);
            }
        }
    }

    const Box& domain = pc.Geom(0).Domain();

    /* This is where workplaces should be assigned */
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();
        auto np = soa.numParticles();

        auto unit_arr = unit_mf[mfi].array();
        auto comm_arr = comm_mf[mfi].array();

        auto Population = demo.Population_d.data();
        auto Start = demo.Start_d.data();
        auto Ndaywork = demo.Ndaywork_d.data();
        auto Ncommunity = demo.Ncommunity;
        auto Nunit = demo.Nunit;

        amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int ip, RandomEngine const& engine) noexcept
            {
                auto from = unit_arr(home_i_ptr[ip], home_j_ptr[ip], 0);

                /* Randomly assign the eligible working-age population */
                unsigned int number = (unsigned int) rint(((Real) Population[from]) / 2000.0);
                unsigned int nwork = (unsigned int) (2000.0 * number * .586); /* 58.6% of population is working-age */
                if (nwork == 0) { return; }

                int age_group = age_group_ptr[ip];
                /* Check working-age population */
                if ((age_group == 2) || (age_group == 3)) {
                    unsigned int irnd = amrex::Random_int(nwork, engine);
                    int to = 0;
                    int comm_to = 0;
                    if (irnd < flow[from][Nunit-1]) {
                        /* Choose a random destination unit */
                        to = 0;
                        while (irnd >= flow[from][to]) { to++; }
                    }

                    /*If from=to unit, 25% EXTRA chance of working in home community*/
                    if ((from == to) && (amrex::Random(engine) < 0.25)) {
                        comm_to = comm_arr(home_i_ptr[ip], home_j_ptr[ip], 0);
                    } else {
                        /* Choose a random community within that destination unit */
                        comm_to = Start[to] + amrex::Random_int(Start[to+1] - Start[to], engine);
                        AMREX_ALWAYS_ASSERT(comm_to < Ncommunity);
                    }

                    IntVect comm_to_iv = domain.atOffset(comm_to);
                    work_i_ptr[ip] = comm_to_iv[0];
                    work_j_ptr[ip] = comm_to_iv[1];

                    constexpr int WG_size = 20;
                    number = (unsigned int) rint( ((Real) Ndaywork[to]) /
                             ((Real) WG_size * (Start[to+1] - Start[to])) );

                    work_nborhood_ptr[ip]=4*(amrex::Random_int(4, engine))+nborhood_ptr[ip];

                    if (number) {
                        workgroup_ptr[ip] = 1 + amrex::Random_int(number, engine);
                    }
                }
            });
        }
    }

    int infect_random_community (AgentContainer& pc, const amrex::iMultiFab& unit_mf,
                                 const amrex::iMultiFab& /*FIPS_mf*/, const amrex::iMultiFab& comm_mf,
                                 const CaseData& /*cases*/, const DemographicData& demo,
                                 int unit, int ninfect) {
        // chose random community in unit
        int ncomms = demo.Start[unit+1] - demo.Start[unit];
        int random_comm;
        if (ParallelDescriptor::IOProcessor()) {
            random_comm = amrex::Random_int(ncomms) + demo.Start[unit];
        }
        ParallelDescriptor::Bcast(&random_comm, 1);

        const Geometry& geom = pc.Geom(0);
        IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        int num_infected = 0;
        for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            amrex::DenseBins<AgentContainer::ParticleType> bins;
            auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
            auto& aos = agents_tile.GetArrayOfStructs();
            auto& soa = agents_tile.GetStructOfArrays();
            const size_t np = aos.numParticles();
            if (np == 0) { continue; }
            auto pstruct_ptr = aos().dataPtr();
            const Box& box = mfi.validbox();

            int ntiles = numTilesInBox(box, true, bin_size);

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, box};
            bins.build(np, pstruct_ptr, ntiles, binner);
            auto inds = bins.permutationPtr();
            auto offsets = bins.offsetsPtr();

            auto status_ptr = soa.GetIntData(IntIdx::age_group).data();
            //auto unit_arr = unit_mf[mfi].array();
            auto comm_arr = comm_mf[mfi].array();
            auto bx = mfi.tilebox();

            Gpu::DeviceScalar<int> num_infected_d(num_infected);
            int* num_infected_p = num_infected_d.dataPtr();
            amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
            {
                int community = comm_arr(i, j, k);
                if (community != random_comm) { return; }

                //int i_cell = community;  // I think this is true...
                Box tbx;
                int i_cell = getTileIndex({AMREX_D_DECL(i, j, k)}, box, true, bin_size, tbx);
                //AMREX_ASSERT(tid == i_cell);
                auto cell_start = offsets[i_cell];
                auto cell_stop  = offsets[i_cell+1];
                int num_this_community = cell_stop - cell_start;
                AMREX_ASSERT(num_this_community > 0);
                AMREX_ASSERT(cell_stop < np);

                if (num_this_community == 0) { return;}

                int ntry = 0;
                int ni = 0;
                unsigned int stop = std::min(cell_start + ninfect, cell_stop);
                for (unsigned int ip = cell_start; ip < stop; ++ip) {
                    int ind = cell_start + amrex::Random_int(num_this_community, engine);
                    auto pindex = inds[ind];
                    if (status_ptr[pindex] == Status::infected
                        || status_ptr[pindex] == Status::immune) {
                        if (++ntry < 100) {
                            --ip;
                        } else {
                            ip += ninfect;
                        }
                    } else {
                        status_ptr[pindex] = Status::infected;
                        ++ni;
                    }
                }
                //amrex::Print() << "infected " << ni << " in " << ntry << " tries in comm " << community << "\n";
                *num_infected_p = ni;
            });

            Gpu::Device::streamSynchronize();
            num_infected += num_infected_d.dataValue();
        }

        ParallelDescriptor::ReduceIntSum(num_infected);
        return num_infected;
    }

    void setInitialCases (AgentContainer& pc, const amrex::iMultiFab& unit_mf,
                          const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& comm_mf,
                          const CaseData& cases, const DemographicData& demo)
    {
        amrex::Vector<int> FIPS_code_to_i(57000, -1);
        for (int i = 0; i < demo.FIPS.size(); ++i) {
            FIPS_code_to_i[demo.FIPS[i]] = i;
        }

        int ntry = 5;
        int ninf = 0;
        for (int ihub = 0; ihub < cases.N_hubs; ++ihub) {
            if (cases.Size_hubs[ihub] > 0) {
                int FIPS = cases.FIPS_hubs[ihub];
                int unit = FIPS_code_to_i[FIPS];
                if (unit > 0) {
                    amrex::Print() << unit << " " << cases.Size_hubs[ihub] << "\n";
                    for (int i = 0; i < cases.Size_hubs[ihub]; i+=5) {
                        if (infect_random_community(pc, unit_mf, FIPS_mf, comm_mf,
                                                    cases, demo, unit, ntry) < ntry) {
                            i--;  // try again
                        } else {
                            ninf += ntry;
                        }
                    }
                    return;
                }
            }
        }
    }

}
}
