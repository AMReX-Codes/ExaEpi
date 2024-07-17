/*! @file AgentContainer.cpp
    \brief Function implementations for #AgentContainer class
*/

#include "AgentContainer.H"

using namespace amrex;

namespace {

    /*! \brief Shuffle the elements of a given vector */
    void randomShuffle (std::vector<int>& vec /*!< Vector to be shuffled */)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(vec.begin(), vec.end(), g);
    }

    /*! \brief
    */
    void compute_initial_distribution (amrex::Vector<int>& cell_pops, /*!< */
                                       amrex::Vector<int>& cell_indices, /*!< */
                                       int ncell /*!< */)
    {
        BL_PROFILE("compute_initial_distribution");

        AMREX_ALWAYS_ASSERT(ncell == 3000); // hard-coded right now

        cell_pops.resize(0);
        cell_pops.resize(ncell*ncell, -1);

        // we compute the initial distribution on Rank 0 and broadcast to all ranks
        if (ParallelDescriptor::IOProcessor())
        {
            int num_pop_bins = 1000;
            amrex::Real log_min_pop = 1.062_rt;
            amrex::Real log_max_pop = 4.0_rt;
            amrex::Vector<amrex::Real> cell_pop_bins_r(num_pop_bins);
            amrex::Vector<amrex::Real> num_cells_per_bin_r(num_pop_bins);

            for (int i = 0; i < cell_pop_bins_r.size(); ++i) {
                cell_pop_bins_r[i] = std::pow(10.0_rt,
                    log_min_pop + i*(log_max_pop - log_min_pop)/(num_pop_bins-1));
                num_cells_per_bin_r[i] = std::pow(cell_pop_bins_r[i], -1.5_rt);
            }

            amrex::Real norm = 0_rt;
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
            amrex::Real border_pop = 0_rt;
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

/*! Add runtime SoA attributes */
void AgentContainer::add_attributes()
{
    const bool communicate_this_comp = true;
    {
        int count(0);
        for (int i = 0; i < m_num_diseases*RealIdxDisease::nattribs; i++) {
            AddRealComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " real-type run-time SoA attibute(s).\n";
    }
    {
        int count(0);
        for (int i = 0; i < m_num_diseases*IntIdxDisease::nattribs; i++) {
            AddIntComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " integer-type run-time SoA attibute(s).\n";
    }
    return;
}

/*! \brief Initialize agents for ExaEpi::ICType::Demo */
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

    int i_RT = IntIdx::nattribs;
    int r_RT = RealIdx::nattribs;
    int n_disease = m_num_diseases;

    GpuArray<int*,ExaEpi::max_num_diseases> status_ptrs, strain_ptrs;
    GpuArray<ParticleReal*,ExaEpi::max_num_diseases> counter_ptrs;
    for (int d = 0; d < n_disease; d++) {
        status_ptrs[d] = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();
        strain_ptrs[d] = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::strain).data();
        counter_ptrs[d] = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
    }

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
            p.pos(0) = idx + 0.5_rt;
            p.pos(1) = idy + 0.5_rt;
            p.id() = i;
            p.cpu() = 0;

            for (int d = 0; d < n_disease; d++) {
                counter_ptrs[d][i] = 0.0_rt;
                strain_ptrs[d][i] = 0;

                if (amrex::Random(engine) < 1e-6) {
                    status_ptrs[d][i] = 1;
                    if (amrex::Random(engine) < 0.3) {
                        strain_ptrs[d][i] = 1;
                    }
                }
            }
        }
    });

    amrex::Print() << "Initial Redistribute... ";

    Redistribute();

    amrex::Print() << "... finished initialization\n";
}

/*! \brief Initialize agents for ExaEpi::ICType::Census

 *  + Define and allocate the following integer MultiFabs:
 *    + num_families: number of families; has 7 components, each component is the
 *      number of families of size (component+1)
 *    + fam_offsets: offset array for each family (i.e., each component of each grid cell), where the
 *      offset is the total number of people before this family while iterating over the grid.
 *    + fam_id: ID array for each family ()i.e., each component of each grid cell, where the ID is the
 *      total number of families before this family while iterating over the grid.
 *  + At each grid cell in each box/tile on each processor:
 *    + Set community number.
 *    + Find unit number for this community; specify that a part of this unit is on this processor;
 *      set unit number, FIPS code, and census tract number at this grid cell (community).
 *    + Set community size: 2000 people, unless this is the last community of a unit, in which case
 *      the remaining people if > 1000 (else 0).
 *    + Compute cumulative distribution (on a scale of 0-1000) of household size ranging from 1 to 7:
 *      initialize with default distributions, then compute from census data if available.
 *    + For each person in this community, generate a random integer between 0 and 1000; based on its
 *      value, assign this person to a household of a certain size (1-7) based on the cumulative
 *      distributions above.
 *  + Compute total number of agents (people), family offsets and IDs over the box/tile.
 *  + Allocate particle container AoS and SoA arrays for the computed number of agents.
 *  + At each grid cell in each box/tile on each processor, and for each component (where component
 *    corresponds to family size):
 *    + Compute percentage of school age kids (kids of age 5-17 as a fraction of total kids - under 5
 *      plus 5-17), if available in census data or set to default (76%).
 *    + For each agent at this grid cell and family size (component):
 *      + Find age group by generating a random integer (0-100) and using default age distributions.
 *        Look at code to see the algorithm for family size > 1.
 *      + Set agent position at the center of this grid cell.
 *      + Initialize status and day counters.
 *      + Set age group and family ID.
 *      + Set home location to current grid cell.
 *      + Initialize work location to current grid cell. Actual work location is set in
 *        ExaEpi::read_workerflow().
 *      + Set neighborhood and work neighborhood values. Actual work neighborhood is set
 *        in ExaEpi::read_workerflow().
 *      + Initialize workgroup to 0. It is set in ExaEpi::read_workerflow().
 *      + If age group is 5-17, assign a school based on neighborhood (#assign_school).
 *  + Copy everything to GPU device.
*/
void AgentContainer::initAgentsCensus (iMultiFab& num_residents,    /*!< Number of residents in each community (grid cell);
                                                                         component 0: age under 5,
                                                                         component 1: age group 5-17,
                                                                         component 2: age group 18-29,
                                                                         component 3: age group 30-64,
                                                                         component 4: age group 65+,
                                                                         component 4: total. */
                                       iMultiFab& unit_mf,          /*!< Unit number of each community */
                                       iMultiFab& FIPS_mf,          /*!< FIPS code (component 0) and
                                                                         census tract number (component 1)
                                                                         of each community */
                                       iMultiFab& comm_mf,          /*!< Community number */
                                       DemographicData& demo        /*!< Demographic data */ )
{
    BL_PROFILE("initAgentsCensus");

    const Box& domain = Geom(0).Domain();

    num_residents.setVal(0);
    unit_mf.setVal(-1);
    FIPS_mf.setVal(-1);
    comm_mf.setVal(-1);

    iMultiFab num_families(num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    iMultiFab fam_offsets (num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    iMultiFab fam_id (num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    num_families.setVal(0);

    auto Nunit = demo.Nunit;
    auto Ncommunity = demo.Ncommunity;
    unit_teacher_counts_d.resize(Nunit, 0);
    /* One can decide to define a iMultifab teachercounts -- but, data locality might be challenging*/
    comm_teacher_counts_total_d.resize(Ncommunity, 0);
    comm_teacher_counts_high_d.resize(Ncommunity, 0);
    comm_teacher_counts_middle_d.resize(Ncommunity, 0);
    comm_teacher_counts_elem3_d.resize(Ncommunity, 0);
    comm_teacher_counts_elem4_d.resize(Ncommunity, 0);
    comm_teacher_counts_daycr_d.resize(Ncommunity, 0);

    amrex::Gpu::DeviceVector<long> student_teacher_ratios_d(student_teacher_ratios.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, student_teacher_ratios.begin(), student_teacher_ratios.end(), student_teacher_ratios_d.begin());

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

        auto ratios = student_teacher_ratios_d.dataPtr();
        auto unit_teacher_counts_d_ptr = unit_teacher_counts_d.data();
        auto comm_teacher_counts_total_d_ptr = comm_teacher_counts_total_d.data();
        auto comm_teacher_counts_high_d_ptr = comm_teacher_counts_high_d.data();
        auto comm_teacher_counts_middle_d_ptr = comm_teacher_counts_middle_d.data();
        auto comm_teacher_counts_elem3_d_ptr = comm_teacher_counts_elem3_d.data();
        auto comm_teacher_counts_elem4_d_ptr = comm_teacher_counts_elem4_d.data();
        auto comm_teacher_counts_daycr_d_ptr = comm_teacher_counts_daycr_d.data();

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
        {
            BL_PROFILE("setFamily_id_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_id[mfi].dataPtr();
            Scan::PrefixSum<int>(ncomp*ncell,
                                 [=] AMREX_GPU_DEVICE (int i) -> int {
                                     return in[i];
                                 },
                                 [=] AMREX_GPU_DEVICE (int i, int const& x) { out[i] = x; },
                                 Scan::Type::exclusive, Scan::retSum);
        }

        auto offset_arr = fam_offsets[mfi].array();
        auto fam_id_arr = fam_id[mfi].array();
        auto& agents_tile = DefineAndReturnParticleTile(0, mfi);
        agents_tile.resize(nagents);
        auto aos = &agents_tile.GetArrayOfStructs()[0];
        auto& soa = agents_tile.GetStructOfArrays();

        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto family_ptr = soa.GetIntData(IntIdx::family).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
        auto school_ptr = soa.GetIntData(IntIdx::school).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;
        int n_disease = m_num_diseases;

        GpuArray<int*,ExaEpi::max_num_diseases> status_ptrs;
        GpuArray<ParticleReal*,ExaEpi::max_num_diseases> counter_ptrs;
        for (int d = 0; d < n_disease; d++) {
            status_ptrs[d] = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();
            counter_ptrs[d] = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
        }

        auto timer_ptr = soa.GetRealData(RealIdx::treatment_timer).data();
        auto dx = ParticleGeom(0).CellSizeArray();
        auto my_proc = ParallelDescriptor::MyProc();

        auto student_counts_arr = student_counts[mfi].array();

        Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (init_agents_nextid)
#endif
        {
            pid = PType::NextID();
            PType::NextID(pid+nagents);
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
            int family_id_start = fam_id_arr(i, j, k, n);
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
            int nborhood = 0;
            for (int ii = 0; ii < num_to_add; ++ii) {
                int ip = start + ii;
                auto& agent = aos[ip];
                int il2 = amrex::Random_int(100, engine);
                if (ii % family_size == 0) {
                    nborhood = amrex::Random_int(4, engine);
                }
                int age_group = -1;

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

                agent.pos(0) = (i + 0.5_rt)*dx[0];
                agent.pos(1) = (j + 0.5_rt)*dx[1];
                agent.id()  = pid+ip;
                agent.cpu() = my_proc;

                for (int d = 0; d < n_disease; d++) {
                    status_ptrs[d][ip] = 0;
                    counter_ptrs[d][ip] = 0.0_rt;
                }
                timer_ptr[ip] = 0.0_rt;
                age_group_ptr[ip] = age_group;
                family_ptr[ip] = family_id_start + (ii / family_size);
                home_i_ptr[ip] = i;
                home_j_ptr[ip] = j;
                work_i_ptr[ip] = i;
                work_j_ptr[ip] = j;
                nborhood_ptr[ip] = nborhood;
                work_nborhood_ptr[ip] = 5*nborhood;
                workgroup_ptr[ip] = 0;

                if (age_group == 0) {
                    school_ptr[ip] = 5; // note - need to handle playgroups
                } else if (age_group == 1) {
                    school_ptr[ip] = assign_school(nborhood, engine);
                } else {
                    school_ptr[ip] = -1;
                }

                // Increment the appropriate student counter based on the school assignment
                if (school_ptr[ip] == SchoolType::elem_3) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::elem_3), 1);
                } else if (school_ptr[ip] == SchoolType::elem_4) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::elem_4), 1);
                } else if (school_ptr[ip] == SchoolType::middle) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::middle), 1);
                } else if (school_ptr[ip] == SchoolType::high) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::high), 1);
                } else if (school_ptr[ip] == SchoolType::day_care) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::day_care), 1);
                }

                if (school_ptr[ip]>0) {amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::total), 1); }

            }
        });

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int comm = comm_arr(i,j,k);

            comm_teacher_counts_high_d_ptr[comm]   = (int)((student_counts_arr(i, j, k, SchoolType::high))     / (ratios[SchoolType::high]));
            comm_teacher_counts_middle_d_ptr[comm] = (int)((student_counts_arr(i, j, k, SchoolType::middle))   / (ratios[SchoolType::middle]));
            comm_teacher_counts_elem3_d_ptr[comm]  = (int)((student_counts_arr(i, j, k, SchoolType::elem_3))   / (ratios[SchoolType::elem_3]));
            comm_teacher_counts_elem4_d_ptr[comm]  = (int)((student_counts_arr(i, j, k, SchoolType::elem_4))   / (ratios[SchoolType::elem_4]));
            comm_teacher_counts_daycr_d_ptr[comm]  = (int)((student_counts_arr(i, j, k, SchoolType::day_care)) / (ratios[SchoolType::day_care]));

            int total = comm_teacher_counts_high_d_ptr[comm]
                      + comm_teacher_counts_middle_d_ptr[comm]
                      + comm_teacher_counts_elem3_d_ptr[comm]
                      + comm_teacher_counts_elem4_d_ptr[comm]
                      + comm_teacher_counts_daycr_d_ptr[comm];
            comm_teacher_counts_total_d_ptr[comm] = total;
            amrex::Gpu::Atomic::AddNoRet(&unit_teacher_counts_d_ptr[unit_arr(i,j,k,0)],total);
        });
    }

    demo.CopyToHostAsync(demo.Unit_on_proc_d, demo.Unit_on_proc);
    amrex::Gpu::streamSynchronize();
}

/*! \brief Send agents on a random walk around the neighborhood

    For each agent, set its position to a random one near its current position
*/
void AgentContainer::moveAgentsRandomWalk ()
{
    BL_PROFILE("AgentContainer::moveAgentsRandomWalk");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
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

/*! \brief Move agents to work

    For each agent, set its position to the work community (IntIdx::work_i, IntIdx::work_j)
*/
void AgentContainer::moveAgentsToWork ()
{
    BL_PROFILE("AgentContainer::moveAgentsToWork");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
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
                p.pos(0) = (work_i_ptr[ip] + 0.5_prt)*dx[0];
                p.pos(1) = (work_j_ptr[ip] + 0.5_prt)*dx[1];
            });
        }
    }

    m_at_work = true;
}

/*! \brief Move agents to home

    For each agent, set its position to the home community (IntIdx::home_i, IntIdx::home_j)
*/
void AgentContainer::moveAgentsToHome ()
{
    BL_PROFILE("AgentContainer::moveAgentsToHome");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
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
                p.pos(0) = (home_i_ptr[ip] + 0.5_prt)*dx[0];
                p.pos(1) = (home_j_ptr[ip] + 0.5_prt)*dx[1];
            });
        }
    }

    m_at_work = false;
}

/*! \brief Move agents randomly

    For each agent, set its position to a random location with a probabilty of 0.01%
*/
void AgentContainer::moveRandomTravel ()
{
    BL_PROFILE("AgentContainer::moveRandomTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
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

/*! \brief Updates disease status of each agent */
void AgentContainer::updateStatus (MFPtrVec& a_disease_stats /*!< Community-wise disease stats tracker */)
{
    BL_PROFILE("AgentContainer::updateStatus");

    m_disease_status.updateAgents(*this, a_disease_stats);
}

/*! \brief Start shelter-in-place */
void AgentContainer::shelterStart ()
{
    BL_PROFILE("AgentContainer::shelterStart");

    amrex::Print() << "Starting shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            auto shelter_compliance = m_shelter_compliance;
            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
                if (amrex::Random(engine) < shelter_compliance) {
                    withdrawn_ptr[i] = 1;
                }
            });
        }
    }
}

/*! \brief Stop shelter-in-place */
void AgentContainer::shelterStop ()
{
    BL_PROFILE("AgentContainer::shelterStop");

    amrex::Print() << "Stopping shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                withdrawn_ptr[i] = 0;
            });
        }
    }
}

/*! \brief Infect agents based on their current status and the computed probability of infection.
    The infection probability is computed in AgentContainer::interactAgentsHomeWork() or
    AgentContainer::interactAgents() */
void AgentContainer::infectAgents ()
{
    BL_PROFILE("AgentContainer::infectAgents");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;
            int n_disease = m_num_diseases;

            for (int d = 0; d < n_disease; d++) {

                auto status_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();

                auto counter_ptr           = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
                auto prob_ptr              = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::prob).data();
                auto incubation_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::incubation_period).data();
                auto infectious_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::infectious_period).data();
                auto symptomdev_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::symptomdev_period).data();

                auto* lparm = d_parm[d];

                amrex::ParallelForRNG( np,
                [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
                {
                    prob_ptr[i] = 1.0_rt - prob_ptr[i];
                    if ( status_ptr[i] == Status::never ||
                         status_ptr[i] == Status::susceptible ) {
                        if (amrex::Random(engine) < prob_ptr[i]) {
                            status_ptr[i] = Status::infected;
                            counter_ptr[i] = 0.0_rt;
                            incubation_period_ptr[i] = amrex::RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                            infectious_period_ptr[i] = amrex::RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                            symptomdev_period_ptr[i] = amrex::RandomNormal(lparm->symptomdev_length_mean, lparm->symptomdev_length_std, engine);
                            return;
                        }
                    }
                });
            }
        }
    }
}

/*! \brief Computes the number of agents with various #Status in each grid cell of the
    computational domain.

    Given a MultiFab with at least 5 x (number of diseases) components that is defined with
    the same box array and distribution mapping as this #AgentContainer, the MultiFab will
    contain (at the end of this function) the following *in each cell*:
    For each disease (d being the disease index):
    + component 5*d+0: total number of agents in this grid cell.
    + component 5*d+1: number of agents that have never been infected (#Status::never)
    + component 5*d+2: number of agents that are infected (#Status::infected)
    + component 5*d+3: number of agents that are immune (#Status::immune)
    + component 5*d+4: number of agents that are susceptible infected (#Status::susceptible)
*/
void AgentContainer::generateCellData (MultiFab& mf /*!< MultiFab with at least 5*m_num_diseases components */) const
{
    BL_PROFILE("AgentContainer::generateCellData");

    const int lev = 0;

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();
    int n_disease = m_num_diseases;

    ParticleToMesh(*this, mf, lev,
        [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                              int i,
                              Array4<Real> const& count)
        {
            auto p = ptd.m_aos[i];
            auto iv = getParticleCell(p, plo, dxi, domain);

            for (int d = 0; d < n_disease; d++) {
                int status = ptd.m_runtime_idata[i0(d)+IntIdxDisease::status][i];
                Gpu::Atomic::AddNoRet(&count(iv, 5*d+0), 1.0_rt);
                if (status != Status::dead) {
                    Gpu::Atomic::AddNoRet(&count(iv, 5*d+status+1), 1.0_rt);
                }
            }
        }, false);
}

/*! \brief Computes the total number of agents with each #Status

    Returns a vector with 5 components corresponding to each value of #Status; each element is
    the total number of agents at a step with the corresponding #Status (in that order).
*/
std::array<Long, 9> AgentContainer::getTotals (const int a_d /*!< disease index */) {
    BL_PROFILE("getTotals");
    amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<int,int,int,int,int,int,int,int,int>> (
                  *this, [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, const int i) noexcept
                  -> amrex::GpuTuple<int,int,int,int,int,int,int,int,int>
              {
                  int s[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                  auto status = ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][i];

                  AMREX_ALWAYS_ASSERT(status >= 0);
                  AMREX_ALWAYS_ASSERT(status <= 4);

                  s[status] = 1;

                  if (status == Status::infected) {  // exposed
                      if (notInfectiousButInfected(i, ptd, a_d)) {
                          s[5] = 1;  // exposed, but not infectious
                      } else { // infectious
                          if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::asymptomatic) {
                              s[6] = 1;  // asymptomatic and will remain so
                          }
                          else if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::presymptomatic) {
                              s[7] = 1;  // asymptomatic but will develop symptoms
                          }
                          else if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::symptomatic) {
                              s[8] = 1;  // Infectious and symptomatic
                          } else {
                              amrex::Abort("how did I get here?");
                          }
                      }
                  }
                  return {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]};
              }, reduce_ops);

    std::array<Long, 9> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r),
                                  amrex::get<4>(r), amrex::get<5>(r), amrex::get<6>(r), amrex::get<7>(r),
                                  amrex::get<8>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 9, ParallelDescriptor::IOProcessorNumber());
    return counts;
}

/*! \brief Interaction and movement of agents during morning commute
 *
 * + Move agents to work
 * + Simulate interactions during morning commute (public transit/carpool/etc ?)
*/
void AgentContainer::morningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::morningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToWork();
}

/*! \brief Interaction and movement of agents during evening commute
 *
 * + Simulate interactions during evening commute (public transit/carpool/etc ?)
 * + Simulate interactions at locations agents may stop by on their way home
 * + Move agents to home
*/
void AgentContainer::eveningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::eveningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    //if (haveInteractionModel(ExaEpi::InteractionNames::grocery_store)) {
    //    m_interactions[ExaEpi::InteractionNames::grocery_store]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToHome();
}

/*! \brief Interaction of agents during day time - work and school */
void AgentContainer::interactDay ( MultiFab& a_mask_behavior /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactDay");
    if (haveInteractionModel(ExaEpi::InteractionNames::work)) {
        m_interactions[ExaEpi::InteractionNames::work]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::school)) {
        m_interactions[ExaEpi::InteractionNames::school]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::nborhood)) {
        m_interactions[ExaEpi::InteractionNames::nborhood]->interactAgents( *this, a_mask_behavior );
    }
}

/*! \brief Interaction of agents during evening (after work) - social stuff */
void AgentContainer::interactEvening ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactEvening");
}

/*! \brief Interaction of agents during nighttime time - at home */
void AgentContainer::interactNight ( MultiFab& a_mask_behavior /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactNight");
    if (haveInteractionModel(ExaEpi::InteractionNames::home)) {
        m_interactions[ExaEpi::InteractionNames::home]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::nborhood)) {
        m_interactions[ExaEpi::InteractionNames::nborhood]->interactAgents( *this, a_mask_behavior );
    }
}
