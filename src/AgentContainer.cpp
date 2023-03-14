#include "AgentContainer.H"

using namespace amrex;

namespace {

    void randomShuffle (std::vector<int>& vec)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(vec.begin(), vec.end(), g);
    }

    void compute_initial_distribution (amrex::Vector<int>& cell_pops,
                                       amrex::Vector<int>& cell_indices, int ncell)
    {
        BL_PROFILE("compute_initial_distribution");

        AMREX_ALWAYS_ASSERT(ncell == 3000); // hard-coded right now

        cell_pops.resize(0);
        cell_pops.resize(ncell*ncell, -1);

        // we compute the initial distribution on Rank 0 and broadcast to all ranks
        if (ParallelDescriptor::IOProcessor())
        {
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
                num_cells_per_bin[i] = static_cast<int>(std::round(num_cells_per_bin_r[i]));
                cell_pop_bins[i] = static_cast<int>(std::round(cell_pop_bins_r[i]));
            }

            int total_cells = 0;
            for (int i = 0; i < num_cells_per_bin.size(); ++i) {
                total_cells += num_cells_per_bin[i];
            }
            num_cells_per_bin[0] += (ncell*ncell - total_cells);

            std::vector<int> perm(ncell*ncell);
            std::iota(perm.begin(), perm.end(), 0);
            randomShuffle(perm);

            Vector<int> offsets(num_pop_bins+1);
            offsets[0] = 0;
            for (int i = 1; i < num_pop_bins+1; ++i) {
                offsets[i] = offsets[i-1] + num_cells_per_bin[i-1];
            }

            for (int i = 0; i < num_pop_bins; ++i) {
                for (int j = offsets[i]; j < offsets[i+1]; ++j) {
                    cell_pops[perm[j]] = cell_pop_bins[i];
                }
            }

            int total_agents = 0;
            for (int i = 0; i < cell_pops.size(); ++i) {
                total_agents += cell_pops[i];
            }
            amrex::Print() << "Total number of agents: " << total_agents << "\n";

            amrex::Print() << "Splitting up population into interior and border\n";
            // we now have a list of populations for each cell. We want 1/3
            // of the population to be within 200 cells of the border. We
            // maintain two separate lists, one for the interior, one for the exterior
            int interior_size = 2600*2600;
            int border_size = ncell*ncell - interior_size;

            // First we sort the vector of cell pops
            std::sort(cell_pops.begin(), cell_pops.end());
            amrex::Real border_pop = 0;
            int i = cell_pops.size()-1;
            std::vector<int> border_ids;
            std::vector<int> interior_ids;
            while ((border_pop < 100e6) && (i >= 0)) {
                amrex::Real pop = cell_pops[i];
                if (amrex::Random() < 0.5) {
                    border_ids.push_back(i);
                    border_pop += pop;
                }
                else {
                    interior_ids.push_back(i);
                }
                --i;
            }

            while (interior_ids.size() < static_cast<std::size_t>(interior_size)) {
                interior_ids.push_back(i);
                --i;
            }

            while (i >= 0) {
                amrex::Real pop = cell_pops[i];
                border_pop += pop;
                border_ids.push_back(i);
                --i;
            }

            // if these conditions are not met, then something has gone wrong with the border pop
            AMREX_ALWAYS_ASSERT(i == -1);
            AMREX_ALWAYS_ASSERT(interior_ids.size() == static_cast<std::size_t>(interior_size));
            AMREX_ALWAYS_ASSERT(border_ids.size() == static_cast<std::size_t>(border_size));

            amrex::Print() << "Population within 200 cells of border is " << border_pop << "\n";

            randomShuffle(border_ids);
            randomShuffle(interior_ids);

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
        } else {
            cell_indices.resize(0);
            cell_indices.resize(ncell*ncell);
        }

        // Broadcast
        ParallelDescriptor::Bcast(&cell_pops[0], cell_pops.size(),
                                  ParallelDescriptor::IOProcessorNumber());
        ParallelDescriptor::Bcast(&cell_indices[0], cell_indices.size(),
                                  ParallelDescriptor::IOProcessorNumber());
    }
}

void AgentContainer::initAgentsDemo (iMultiFab& /*num_residents*/,
                                     iMultiFab& /*unit_mf*/,
                                     iMultiFab& /*FIPS_mf*/,
                                     iMultiFab& /*comm_mf*/,
                                     DemographicData& /*demo*/)
{
    BL_PROFILE("AgentContainer::initAgentsDemo");

    int ncell = 3000;
    Vector<int> cell_pops;
    Vector<int> cell_indices;

    compute_initial_distribution(cell_pops, cell_indices, ncell);

    // now each rank will only actually add a subset of the particles
    int ibegin, iend;
    {
        int myproc = ParallelDescriptor::MyProc();
        int nprocs = ParallelDescriptor::NProcs();
        int navg = ncell*ncell/nprocs;
        int nleft = ncell*ncell - navg * nprocs;
        if (myproc < nleft) {
            ibegin = myproc*(navg+1);
            iend = ibegin + navg+1;
        } else {
            ibegin = myproc*navg + nleft;
            iend = ibegin + navg;
        }
    }
    std::size_t ncell_this_rank = iend-ibegin;

    std::size_t np_this_rank = 0;
    for (int i = 0; i < ncell*ncell; ++i) {
        if ((i < ibegin) || (i >= iend)) {
            cell_pops[cell_indices[i]] = 0;
        } else {
            np_this_rank += cell_pops[cell_indices[i]];
        }
    }

    // copy data to GPU
    amrex::Gpu::DeviceVector<int> cell_pops_d(cell_pops.size());
    amrex::Gpu::DeviceVector<int> cell_offsets_d(cell_pops.size()+1);
    Gpu::copy(Gpu::hostToDevice, cell_pops.begin(), cell_pops.end(),
              cell_pops_d.begin());
    Gpu::exclusive_scan(cell_pops_d.begin(), cell_pops_d.end(), cell_offsets_d.begin());

    amrex::Gpu::DeviceVector<int> cell_indices_d(cell_indices.size());
    Gpu::copy(Gpu::hostToDevice, cell_indices.begin(), cell_indices.end(), cell_indices_d.begin());

    // Fill in particle data in each cell
    auto& ptile = DefineAndReturnParticleTile(0, 0, 0);
    ptile.resize(np_this_rank);

    auto& soa   = ptile.GetStructOfArrays();
    auto& aos   = ptile.GetArrayOfStructs();
    auto pstruct_ptr = aos().data();
    auto status_ptr = soa.GetIntData(IntIdx::status).data();
    auto strain_ptr = soa.GetIntData(IntIdx::strain).data();
    auto timer_ptr = soa.GetRealData(RealIdx::timer).data();

    auto cell_offsets_ptr = cell_offsets_d.data();
    auto cell_indices_ptr = cell_indices_d.data();

    amrex::ParallelForRNG( ncell_this_rank,
    [=] AMREX_GPU_DEVICE (int i_this_rank, RandomEngine const& engine) noexcept
    {
        int cell_id = i_this_rank + ibegin;
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
            strain_ptr[i] = 0;

            if (amrex::Random(engine) < 2e-8) {
                status_ptr[i] = 1;
                if (amrex::Random(engine) < 0.3) {
                    strain_ptr[i] = 1;
                }
                timer_ptr[i] = 5.0*24;
            }
        }
    });

    amrex::Print() << "Initial Redistribute... ";

    Redistribute();

    amrex::Print() << "... finished initialization\n";
}

void AgentContainer::initAgentsCensus (iMultiFab& num_residents,
                                       iMultiFab& unit_mf,
                                       iMultiFab& FIPS_mf,
                                       iMultiFab& comm_mf,
                                       DemographicData& demo)
{
    BL_PROFILE("initAgentsCensus");

    using AgentType = ParticleType;

    const Box& domain = Geom(0).Domain();

    num_residents.setVal(0);
    unit_mf.setVal(-1);
    FIPS_mf.setVal(-1);
    comm_mf.setVal(-1);

    iMultiFab num_families(num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    iMultiFab fam_offsets (num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    num_families.setVal(0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto unit_arr = unit_mf[mfi].array();
        auto FIPS_arr = FIPS_mf[mfi].array();
        auto comm_arr = comm_mf[mfi].array();
        auto nf_arr = num_families[mfi].array();
        auto nr_arr = num_residents[mfi].array();

        auto unit_on_proc = demo.Unit_on_proc_d.data();
        auto Start = demo.Start_d.data();
        auto FIPS = demo.FIPS_d.data();
        auto Tract = demo.Tract_d.data();
        auto Population = demo.Population_d.data();

        auto H1 = demo.H1_d.data();
        auto H2 = demo.H2_d.data();
        auto H3 = demo.H3_d.data();
        auto H4 = demo.H4_d.data();
        auto H5 = demo.H5_d.data();
        auto H6 = demo.H6_d.data();
        auto H7 = demo.H7_d.data();

        auto N5  = demo.N5_d.data();
        auto N17 = demo.N17_d.data();
        //auto N29 = demo.N29_d.data();
        //auto N64 = demo.N64_d.data();
        //auto N65plus = demo.N65plus_d.data();

        auto Ncommunity = demo.Ncommunity;

        auto bx = mfi.tilebox();
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            int community = (int) domain.index(IntVect(AMREX_D_DECL(i, j, k)));
            if (community >= Ncommunity) { return; }
            comm_arr(i, j, k) = community;

            int unit = 0;
            while (community >= Start[unit+1]) { unit++; }
            unit_on_proc[unit] = 1;
            unit_arr(i, j, k) = unit;
            FIPS_arr(i, j, k, 0) = FIPS[unit];
            FIPS_arr(i, j, k, 1) = Tract[unit];

            int community_size;
            if (Population[unit] < (1000 + 2000*(community - Start[unit]))) {
                community_size = 0;  /* Don't set up any residents; workgroup-only */
            }
            else {
                community_size = 2000;   /* Standard 2000-person community */
            }

            int p_hh[7] = {330, 670, 800, 900, 970, 990, 1000};
            int num_hh = H1[unit] + H2[unit] + H3[unit] +
                H4[unit] + H5[unit] + H6[unit] + H7[unit];
            if (num_hh) {
                p_hh[0] = 1000 * H1[unit] / num_hh;
                p_hh[1] = 1000* (H1[unit] + H2[unit]) / num_hh;
                p_hh[2] = 1000* (H1[unit] + H2[unit] + H3[unit]) / num_hh;
                p_hh[3] = 1000* (H1[unit] + H2[unit] + H3[unit] + H4[unit]) / num_hh;
                p_hh[4] = 1000* (H1[unit] + H2[unit] + H3[unit] +
                                 H4[unit] + H5[unit]) / num_hh;
                p_hh[5] = 1000* (H1[unit] + H2[unit] + H3[unit] +
                                 H4[unit] + H5[unit] + H6[unit]) / num_hh;
                p_hh[6] = 1000;
            }

            int npeople = 0;
            while (npeople < community_size + 1) {
                int il  = amrex::Random_int(1000, engine);

                int family_size = 1;
                while (il > p_hh[family_size]) { ++family_size; }
                AMREX_ASSERT(family_size > 0);
                AMREX_ASSERT(family_size <= 7);

                nf_arr(i, j, k, family_size-1) += 1;
                npeople += family_size;
            }

            AMREX_ASSERT(npeople == nf_arr(i, j, k, 0) +
                         2*nf_arr(i, j, k, 1) +
                         3*nf_arr(i, j, k, 2) +
                         4*nf_arr(i, j, k, 3) +
                         5*nf_arr(i, j, k, 4) +
                         6*nf_arr(i, j, k, 5) +
                         7*nf_arr(i, j, k, 6));

            nr_arr(i, j, k, 5) = npeople;
        });

        int nagents;
        int ncomp = num_families[mfi].nComp();
        int ncell = num_families[mfi].numPts();
        {
            BL_PROFILE("setPopulationCounts_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_offsets[mfi].dataPtr();
            nagents = Scan::PrefixSum<int>(ncomp*ncell,
                            [=] AMREX_GPU_DEVICE (int i) -> int {
                                int comp = i / ncell;
                                return (comp+1)*in[i];
                            },
                            [=] AMREX_GPU_DEVICE (int i, int const& x) { out[i] = x; },
                                               Scan::Type::exclusive, Scan::retSum);

        }

        auto offset_arr = fam_offsets[mfi].array();
        auto& agents_tile = GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
        agents_tile.resize(nagents);
        auto aos = &agents_tile.GetArrayOfStructs()[0];
        auto& soa = agents_tile.GetStructOfArrays();

        auto status_ptr = soa.GetIntData(IntIdx::status).data();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();

        auto timer_ptr = soa.GetRealData(RealIdx::timer).data();
        auto dx = ParticleGeom(0).CellSizeArray();
        auto my_proc = ParallelDescriptor::MyProc();

        Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (init_agents_nextid)
#endif
        {
            pid = AgentType::NextID();
            AgentType::NextID(pid+nagents);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<Long>(pid + nagents) < LastParticleID,
            "Error: overflow on agent id numbers!");

        amrex::ParallelForRNG(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, amrex::RandomEngine const& engine) noexcept
        {
            int nf = nf_arr(i, j, k, n);
            if (nf == 0) return;

            int unit = unit_arr(i, j, k);
            int community = comm_arr(i, j, k);
            int family_size = n + 1;
            int num_to_add = family_size * nf;

            int community_size;
            if (Population[unit] < (1000 + 2000*(community - Start[unit]))) {
                community_size = 0;  /* Don't set up any residents; workgroup-only */
            }
            else {
                community_size = 2000;   /* Standard 2000-person community */
            }

            int p_schoolage = 0;
            if (community_size) {  // Only bother for residential communities
                if (N5[unit] + N17[unit]) {
                    p_schoolage = 100*N17[unit] / (N5[unit] + N17[unit]);
                }
                else {
                    p_schoolage = 76;
                }
            }

            int start = offset_arr(i, j, k, n);
            for (int ip = start; ip < start + num_to_add; ++ip) {
                auto& agent = aos[ip];
                int il2 = amrex::Random_int(100, engine);

                int age_group;
                if (family_size == 1) {
                    if (il2 < 28) { age_group = 4; }      /* single adult age 65+   */
                    else if (il2 < 68) { age_group = 3; } /* age 30-64 (ASSUME 40%) */
                    else { age_group = 2; }               /* single adult age 19-29 */
                    nr_arr(i, j, k, age_group) += 1;
                } else if (family_size == 2) {
                    if (il2 == 0) {
                        /* 1% probability of one parent + one child */
                        int il3 = amrex::Random_int(100, engine);
                        if (il3 < 2) { age_group = 4; }        /* one parent, age 65+ */
                        else if (il3 < 62) { age_group = 3; }  /* one parent 30-64 (ASSUME 60%) */
                        else { age_group = 2; }                /* one parent 19-29 */
                        nr_arr(i, j, k, age_group) += 1;
                        if (((int) amrex::Random_int(100, engine)) < p_schoolage) {
                            age_group = 1; /* 22.0% of total population ages 5-18 */
                        } else {
                            age_group = 0;   /* 6.8% of total population ages 0-4 */
                        }
                        nr_arr(i, j, k, age_group) += 1;
                    } else {
                        /* 2 adults, 28% over 65 (ASSUME both same age group) */
                        if (il2 < 28) { age_group = 4; }      /* single adult age 65+ */
                        else if (il2 < 68) { age_group = 3; } /* age 30-64 (ASSUME 40%) */
                        else { age_group = 2; }               /* single adult age 19-29 */
                        nr_arr(i, j, k, age_group) += 2;
                    }
                }

                if (family_size > 2) {
                    /* ASSUME 2 adults, of the same age group */
                    if (il2 < 2) { age_group = 4; }  /* parents are age 65+ */
                    else if (il2 < 62) { age_group = 3; }  /* parents 30-64 (ASSUME 60%) */
                    else { age_group = 2; }  /* parents 19-29 */
                    nr_arr(i, j, k, age_group) += 2;

                    /* Now pick the children's age groups */
                    for (int nc = 2; nc < family_size; ++nc) {
                        if (((int) amrex::Random_int(100, engine)) < p_schoolage) {
                            age_group = 1; /* 22.0% of total population ages 5-18 */
                        } else {
                            age_group = 0;   /* 6.8% of total population ages 0-4 */
                        }
                        nr_arr(i, j, k, age_group) += 1;
                    }
                }

                agent.pos(0) = (i + 0.5)*dx[0];
                agent.pos(1) = (j + 0.5)*dx[1];
                agent.id()  = pid+ip;
                agent.cpu() = my_proc;

                status_ptr[ip] = 0;
                timer_ptr[ip] = 0.0;
                age_group_ptr[ip] = age_group;
                home_i_ptr[ip] = i;
                home_j_ptr[ip] = j;
                work_i_ptr[ip] = i;
                work_j_ptr[ip] = j;

                if (amrex::Random(engine) < 2e-8) {
                    status_ptr[ip] = 1;
                    timer_ptr[ip] = 5.0*24;
                }
            }
        });
    }

    demo.CopyToHostAsync(demo.Unit_on_proc_d, demo.Unit_on_proc);
    amrex::Gpu::streamSynchronize();
}

void AgentContainer::moveAgentsRandomWalk ()
{
    BL_PROFILE("AgentContainer::moveAgentsRandomWalk");

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

void AgentContainer::moveAgentsToWork ()
{
    BL_PROFILE("AgentContainer::moveAgentsToWork");

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

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                ParticleType& p = pstruct[ip];
                p.pos(0) = (work_i_ptr[ip] + 0.5)*dx[0];
                p.pos(1) = (work_j_ptr[ip] + 0.5)*dx[1];
            });
        }
    }
}

void AgentContainer::moveAgentsToHome ()
{
    BL_PROFILE("AgentContainer::moveAgentsToHome");

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

            auto& soa = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                ParticleType& p = pstruct[ip];
                p.pos(0) = (home_i_ptr[ip] + 0.5)*dx[0];
                p.pos(1) = (home_j_ptr[ip] + 0.5)*dx[1];
            });
        }
    }
}

void AgentContainer::moveRandomTravel ()
{
    BL_PROFILE("AgentContainer::moveRandomTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
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
                if ( (status_ptr[i] == Status::never) ||
                     (status_ptr[i] == Status::susceptible) ) {
                    return;
                }
                else if (status_ptr[i] == Status::infected) {
                    if (timer_ptr[i] == 0.0) {
                        status_ptr[i] = Status::immune;
                        timer_ptr[i] = 6*30*24; // 6 months in hours
                    } else {
                        timer_ptr[i] -= 1.0;
                    }
                }
                else if (status_ptr[i] == Status::immune) {
                    if (timer_ptr[i] == 0.0) {
                        status_ptr[i] = Status::susceptible;
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
            auto strain_ptr = soa.GetIntData(IntIdx::strain).data();
            auto timer_ptr = soa.GetRealData(RealIdx::timer).data();

            amrex::ParallelForRNG( bins.numBins(),
            [=] AMREX_GPU_DEVICE (int i_cell, amrex::RandomEngine const& engine) noexcept
            {
                auto cell_start = offsets[i_cell];
                auto cell_stop  = offsets[i_cell+1];

                // compute the number of infected in this cell
                int num_infected[2] = {0, 0};
                for (unsigned int i = cell_start; i < cell_stop; ++i) {
                    auto pindex = inds[i];
                    if (status_ptr[pindex] == 1) {
                        ++num_infected[strain_ptr[pindex]];
                    }
                }

                // second pass - infection prob is propto num_infected
                for (unsigned int i = cell_start; i < cell_stop; ++i) {
                    auto pindex = inds[i];
                    if ( (status_ptr[pindex] != Status::infected)
                         && (status_ptr[pindex] != Status::immune)) {
                        if (amrex::Random(engine) < 0.001*num_infected[0]) {
                            strain_ptr[pindex] = 0;
                            status_ptr[pindex] = Status::infected;
                            timer_ptr[pindex] = 5.0*24; // 5 days in hours
                        } else if (amrex::Random(engine) < 0.002*num_infected[1]) {
                            strain_ptr[pindex] = 1;
                            status_ptr[pindex] = Status::infected;
                            timer_ptr[pindex] = 5.0*24; // 5 days in hours
                        }
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
            if (status == Status::never) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 1), 1.0_rt);
            }
            else if (status == Status::infected) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 2), 1.0_rt);
            }
            else if (status == Status::immune) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 3), 1.0_rt);
            }
            else if (status == Status::susceptible) {
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
                  return {s[0], s[1], s[2], s[3]};
              }, reduce_ops);

    Long counts[4] = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 4, ParallelDescriptor::IOProcessorNumber());
    amrex::Print() << "Never infected: " << counts[0] << "\n";
    amrex::Print() << "Infected: " << counts[1] << "\n";
    amrex::Print() << "Immune: " << counts[2] << "\n";
    amrex::Print() << "Previously infected: " << counts[3] << "\n";
}
