#include "AgentContainer.H"

using namespace amrex;

void AgentContainer::initAgents ()
{
    BL_PROFILE("AgentContainer::initAgents");

    int ncell = 3000;

    int num_pop_bins = 1000;
    amrex::Real log_min_pop = 1.062;
    amrex::Real log_max_pop = 4.0;
    amrex::Vector<amrex::Real> cell_pop_bins_r(num_pop_bins);
    amrex::Vector<amrex::Real> num_cells_per_bin_r(num_pop_bins);

    for (int i = 0; i < cell_pop_bins_r.size(); ++i) {
        cell_pop_bins_r[i] = std::pow(10.0,
            log_min_pop + i*(log_max_pop - log_min_pop)/(num_pop_bins-1));
        num_cells_per_bin_r[i] = std::pow(cell_pop_bins_r[i], -1.5);
    }

    amrex::Real norm = 0;
    for (int i = 0; i < num_cells_per_bin_r.size(); ++i) {
        norm += num_cells_per_bin_r[i];
    }

    amrex::Vector<int> cell_pop_bins(num_pop_bins);
    amrex::Vector<int> num_cells_per_bin(num_pop_bins);
    for (int i = 0; i < num_cells_per_bin.size(); ++i) {
        num_cells_per_bin_r[i] *= (ncell*ncell/norm);
        num_cells_per_bin[i] = std::round(num_cells_per_bin_r[i]);
        cell_pop_bins[i] = std::round(cell_pop_bins_r[i]);
    }

    int total_cells = 0;
    for (int i = 0; i < num_cells_per_bin.size(); ++i) {
        total_cells += num_cells_per_bin[i];
    }
    num_cells_per_bin[0] += (ncell*ncell - total_cells);

    int total_agents = 0;
    for (int i = 0; i < num_cells_per_bin.size(); ++i) {
        total_agents += cell_pop_bins[i]*num_cells_per_bin[i];
    }
    amrex::Print() << "Total number of agents: " << total_agents << "\n";

    std::vector<int> perm(ncell*ncell);
    std::iota(perm.begin(), perm.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm.begin(), perm.end(), g);

    Vector<int> offsets(num_pop_bins+1);
    offsets[0] = 0;
    for (int i = 1; i < num_pop_bins+1; ++i) {
        offsets[i] = offsets[i-1] + num_cells_per_bin[i-1];
    }

    Vector<int> cell_pops(ncell*ncell, -1);
    for (int i = 0; i < num_pop_bins; ++i) {
        for (int j = offsets[i]; j < offsets[i+1]; ++j) {
            cell_pops[perm[j]] = cell_pop_bins[i];
        }
    }

    amrex::Gpu::DeviceVector<int> cell_pops_d(cell_pops.size());
    amrex::Gpu::DeviceVector<int> cell_offsets_d(cell_pops.size()+1);
    Gpu::copy(Gpu::hostToDevice, cell_pops.begin(), cell_pops.end(), cell_pops_d.begin());
    Gpu::exclusive_scan(cell_pops_d.begin(), cell_pops_d.end(), cell_offsets_d.begin());

    // Fill in particle data in each cell
    auto& ptile = DefineAndReturnParticleTile(0, 0, 0);
    ptile.resize(total_agents);

    auto& soa   = ptile.GetStructOfArrays();
    auto& aos   = ptile.GetArrayOfStructs();
    auto pstruct_ptr = aos().data();
    auto status_ptr = soa.GetIntData(IntIdx::status).data();
    auto timer_ptr = soa.GetIntData(RealIdx::timer).data();

    auto cell_offsets_ptr = cell_offsets_d.data();

    amrex::Print() << "About to fill data \n";
    
    amrex::ParallelForRNG( ncell * ncell,
    [=] AMREX_GPU_DEVICE (int cell_id, RandomEngine const& engine) noexcept
    {
        int cell_start = cell_offsets_ptr[cell_id];
        int cell_stop = cell_offsets_ptr[cell_id+1];
        
        int idx = cell_id % ncell;
        int idy = cell_id / ncell;

        for (int i = cell_start; i < cell_stop; ++i) {
            auto& p = pstruct_ptr[i];
            p.pos(0) = idx + 0.5;
            p.pos(1) = idy + 0.5;
            p.id() = i;
            p.cpu() = 0;

            timer_ptr[i] = 0.0;

            if (amrex::Random(engine) < 0.05) {
                status_ptr[i] = 1;
            }
        }
    });

    amrex::Print() << "Finished filling data \n";
    
    Redistribute();

    amrex::Print() << "finished initialization \n";
}

void AgentContainer::moveAgents ()
{
    BL_PROFILE("AgentContainer::moveAgents");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                ParticleType& p = pstruct[i];

                p.pos(0) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[0]);
                p.pos(1) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[1]);
            });
        }
    }
}

void AgentContainer::interactAgents ()
{
    BL_PROFILE("AgentContainer::interactAgents");

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < numLevels(); ++lev)
    {
        const Geometry& geom = Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            amrex::DenseBins<ParticleType> bins;
            auto& ptile = ParticlesAt(lev, mfi);
            auto& aos   = ptile.GetArrayOfStructs();
            const size_t np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            const Box& box = mfi.validbox();

            int ntiles = numTilesInBox(box, true, bin_size);

            bins.build(np, pstruct_ptr, ntiles, GetParticleBin{plo, dxi, domain, bin_size, box});
            auto inds = bins.permutationPtr();
            auto offsets = bins.offsetsPtr();

            auto& soa   = ptile.GetStructOfArrays();
            auto d_ptr = soa.GetIntData(IntIdx::status).data();

            amrex::ParallelForRNG( bins.numBins(),
            [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
            {
                auto cell_start = offsets[i_cell];
                auto cell_stop  = offsets[i_cell+1];

                int num_infected = 0;
                for (unsigned int i = cell_start; i < cell_stop; ++i) {
                    auto pindex = inds[i];
                    if (d_ptr[pindex] == 1) { ++num_infected; }
                }

                for (unsigned int i = cell_start; i < cell_stop; ++i) {
                    auto pindex = inds[i];
                    if ( (d_ptr[pindex] != 1) && (amrex::Random(engine) < 0.0001*num_infected)) {
                        d_ptr[pindex] = 1;
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
    }
}
