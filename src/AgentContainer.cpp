#include "AgentContainer.H"

using namespace amrex;

namespace {
    void randomShuffle (std::vector<int>& vec) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(vec.begin(), vec.end(), g);
    }
}

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
    randomShuffle(perm);

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

    amrex::Print() << "Splitting up population into interior and border \n";
    // we now have a list of populations for each cell. We want 1/3
    // of the population to be within 200 cells of the border. We
    // maintain two separate lists, one for the interior, one for the exterior
    int interior_size = 2600*2600;
    int border_size = ncell*ncell - interior_size;

    // First we sort the vector of cell pops
    Vector<int> sorted_cell_pops(cell_pops);
    std::sort(sorted_cell_pops.begin(), sorted_cell_pops.end());
    amrex::Real border_pop = 0;
    int i = sorted_cell_pops.size()-1;
    std::vector<int> border_ids;
    std::vector<int> interior_ids;
    while ((border_pop < 100e6) && (i >= 0)) {
        amrex::Real pop = sorted_cell_pops[i];
        if (amrex::Random() < 0.5) {
            border_ids.push_back(i);
            border_pop += pop;
        }
        else {
            interior_ids.push_back(i);
        }
        --i;
    }

    while (interior_ids.size() < interior_size) {
        interior_ids.push_back(i);
        --i;
    }
    
    while (i >= 0) {
        amrex::Real pop = sorted_cell_pops[i];
        border_pop += pop;
        border_ids.push_back(i);
        --i;
    }

    // if these conditions are not met, then something has gone wrong with the border pop
    AMREX_ALWAYS_ASSERT(i == -1);
    AMREX_ALWAYS_ASSERT(interior_ids.size() == interior_size);
    AMREX_ALWAYS_ASSERT(border_ids.size() == border_size);

    amrex::Print() << "Population within 200 cells of border is " << border_pop << "\n";

    randomShuffle(border_ids);
    randomShuffle(interior_ids);
    
    std::vector<int> cell_indices;
    for (int cell_id = 0; cell_id < ncell*ncell; ++cell_id) {
        int idx = cell_id % ncell;
        int idy = cell_id / ncell;
        if ((idx < 200) || (idx >= 2800) || (idy < 200) || (idy >= 2800)) {
            cell_indices.push_back(border_ids.back());
            border_ids.pop_back();
        } else {
            cell_indices.push_back(interior_ids.back());
            interior_ids.pop_back();
        }
    }
    AMREX_ALWAYS_ASSERT(interior_ids.size() == 0);
    AMREX_ALWAYS_ASSERT(border_ids.size() == 0);
    
    amrex::Gpu::DeviceVector<int> cell_pops_d(cell_pops.size());
    amrex::Gpu::DeviceVector<int> cell_offsets_d(cell_pops.size()+1);
    Gpu::copy(Gpu::hostToDevice, sorted_cell_pops.begin(), sorted_cell_pops.end(),
              cell_pops_d.begin());
    Gpu::exclusive_scan(cell_pops_d.begin(), cell_pops_d.end(), cell_offsets_d.begin());

    amrex::Gpu::DeviceVector<int> cell_indices_d(cell_indices.size());
    Gpu::copy(Gpu::hostToDevice, cell_indices.begin(), cell_indices.end(), cell_indices_d.begin());
    
    // Fill in particle data in each cell
    auto& ptile = DefineAndReturnParticleTile(0, 0, 0);
    ptile.resize(total_agents);

    auto& soa   = ptile.GetStructOfArrays();
    auto& aos   = ptile.GetArrayOfStructs();
    auto pstruct_ptr = aos().data();
    auto status_ptr = soa.GetIntData(IntIdx::status).data();
    auto timer_ptr = soa.GetRealData(RealIdx::timer).data();

    auto cell_offsets_ptr = cell_offsets_d.data();
    auto cell_indices_ptr = cell_indices_d.data();

    amrex::Print() << "About to fill data \n";
    
    amrex::ParallelForRNG( ncell * ncell,
    [=] AMREX_GPU_DEVICE (int cell_id, RandomEngine const& engine) noexcept
    {
        int ind = cell_indices_ptr[cell_id];

        int cell_start = cell_offsets_ptr[ind];
        int cell_stop = cell_offsets_ptr[ind+1];
        
        int idx = cell_id % ncell;
        int idy = cell_id / ncell;

        for (int i = cell_start; i < cell_stop; ++i) {
            auto& p = pstruct_ptr[i];
            p.pos(0) = idx + 0.5;
            p.pos(1) = idy + 0.5;
            p.id() = i;
            p.cpu() = 0;

            timer_ptr[i] = 0.0;

            if (amrex::Random(engine) < 2e-8) {
                status_ptr[i] = 1;
                timer_ptr[i] = 5.0*24;
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

void AgentContainer::moveRandomTravel ()
{
    BL_PROFILE("AgentContainer::moveRandomTravel");

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

                if (amrex::Random(engine) < 0.0001) {
                    p.pos(0) = 3000*amrex::Random(engine);
                    p.pos(1) = 3000*amrex::Random(engine);
                }
            });
        }
    }
}

void AgentContainer::updateStatus ()
{
    BL_PROFILE("AgentContainer::updateStatus");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto timer_ptr = soa.GetRealData(RealIdx::timer).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                // enum would be good here
                if ( (status_ptr[i] == 0) || (status_ptr[i] == 3) ) {
                    return;
                }
                else if (status_ptr[i] == 1) { // infected 
                    if (timer_ptr[i] == 0.0) { 
                        status_ptr[i] = 2;
                        timer_ptr[i] = 6*30*24; // 6 months in hours
                    } else {
                        timer_ptr[i] -= 1.0;
                    }
                }
                else if (status_ptr[i] == 2) { // immune
                    if (timer_ptr[i] == 0.0) {
                        status_ptr[i] = 3;
                    } else {
                        timer_ptr[i] -= 1.0;
                    }
                }
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
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto timer_ptr = soa.GetRealData(RealIdx::timer).data();

            amrex::ParallelForRNG( bins.numBins(),
            [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
            {
                auto cell_start = offsets[i_cell];
                auto cell_stop  = offsets[i_cell+1];

                // compute the number of infected in this cell
                int num_infected = 0;
                for (unsigned int i = cell_start; i < cell_stop; ++i) {
                    auto pindex = inds[i];
                    if (status_ptr[pindex] == 1) { ++num_infected; }
                }

                // second pass - infection prob is propto num_infected
                for (unsigned int i = cell_start; i < cell_stop; ++i) {
                    auto pindex = inds[i];
                    if ( (status_ptr[pindex] != 1) // not currently infected
                      && (status_ptr[pindex] != 2) // not immune
                      && (amrex::Random(engine) < 0.001*num_infected)) {
                        status_ptr[pindex] = 1;
                        timer_ptr[pindex] = 5.0*24; // 5 days in hours
                    }
                }
            });
            amrex::Gpu::synchronize();
        }
    }
}

void AgentContainer::generateCellData (MultiFab& mf)
{
    BL_PROFILE("AgentContainer::generateCellData");

    const int lev = 0;

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();
    amrex::ParticleToMesh(*this, mf, lev,
        [=] AMREX_GPU_DEVICE (const SuperParticleType& p,
                              amrex::Array4<amrex::Real> const& count)
        {
            int status = p.idata(0);
            auto iv = getParticleCell(p, plo, dxi, domain);
            amrex::Gpu::Atomic::AddNoRet(&count(iv, 0), 1.0_rt);
            if (status == 0) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 1), 1.0_rt);
            }
            else if (status == 1) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 2), 1.0_rt);
            }
            else if (status == 2) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 3), 1.0_rt);
            }
            else if (status == 3) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 4), 1.0_rt);
            }
        }, false);
}

void AgentContainer::printTotals () {
    BL_PROFILE("printTotals");
    amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<int,int,int,int>> (
                  *this, [=] AMREX_GPU_DEVICE (const SuperParticleType& p) noexcept
                  -> amrex::GpuTuple<int,int,int,int>
              {
                  int s[4] = {0, 0, 0, 0};
                  s[p.idata(IntIdx::status)] = 1;
                  return {s[0], s[1], s[2], s[4]};
              }, reduce_ops);
    amrex::Print() << "Never infected: " << amrex::get<0>(r) << "\n";
    amrex::Print() << "Infected: " << amrex::get<1>(r) << "\n";
    amrex::Print() << "Immune: " << amrex::get<2>(r) << "\n";
    amrex::Print() << "Previously infected: " << amrex::get<3>(r) << "\n";
}
