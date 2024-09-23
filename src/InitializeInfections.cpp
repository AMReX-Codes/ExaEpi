/*! @file InitializeInfections.cpp
*/

#include <AMReX_ParticleUtil.H>

#include "InitializeInfections.H"


using namespace amrex;
using namespace ExaEpi;


/*! \brief Infect agents in a random community in a given unit and return the total
    number of agents infected

    + Choose a random community in the given unit.
    + For each box on each processor:
        + Create bins of agents if not already created (see #amrex::GetParticleBin, #amrex::DenseBins):
        + The bin size is 1 cell.
        + #amrex::GetParticleBin maps a particle to its bin index.
        + amrex::DenseBins::build() creates the bin-sorted array of particle indices and
            the offset array for each bin (where the offset of a bin is its starting location.
        + For each grid cell: if the community at this cell is the randomly chosen community,
        + Get bin index and the agent (particle) indices in this bin.
        + Choose a random agent in the bin; if the agent is already infected, move on, else
            infect the agent. Increment the counter variables for number of infections.
            (See the code for nuances in this step.)
    + Sum up number of infected agents over all processors and return that value.
*/
static int infect_random_community (AgentContainer& pc, /*!< Agent container (particle container)*/
                                    int ncomms,   /*!< Number of communities*/
                                    const Vector<int> &unit_community_start, /*!< Start community number for each unit */
                                    iMultiFab &unit_mf,
                                    iMultiFab &comm_mf,
                                    std::map<std::pair<int, int>,
                                    DenseBins<AgentContainer::ParticleType> >& bin_map, /*!< Map of dense bins with agents */
                                    int unit, /*!< Unit number to infect */
                                    const int d_idx, /*!< Disease index */
                                    int ninfect, /*!< Target number of agents to infect */
                                    const bool fast_bin /*!< Use GPU binning - fast but non-deterministic */  ) {

    // chose random community
    int comm_offset = 0;
    if (unit > 0) {
        ncomms = unit_community_start[unit + 1] - unit_community_start[unit];
        comm_offset = unit_community_start[unit];
    }

    int random_comm = -1;
    if (ParallelDescriptor::IOProcessor()) {
        random_comm = Random_int(ncomms) + comm_offset;
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
        DenseBins<AgentContainer::ParticleType>& bins = bin_map[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& aos = agents_tile.GetArrayOfStructs();
        auto& soa = agents_tile.GetStructOfArrays();
        const size_t np = aos.numParticles();
        if (np == 0) { continue; }
        auto pstruct_ptr = aos().dataPtr();
        const Box& box = mfi.validbox();

        int ntiles = numTilesInBox(box, true, bin_size);

        auto binner = GetParticleBin{plo, dxi, domain, bin_size, box};
        if (bins.numBins() < 0) {
            if (fast_bin) bins.build(BinPolicy::GPU, np, pstruct_ptr, ntiles, binner);
            else bins.build(BinPolicy::Serial, np, pstruct_ptr, ntiles, binner);
        }
        auto inds = bins.permutationPtr();
        auto offsets = bins.offsetsPtr();

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;

        auto status_ptr = soa.GetIntData(i_RT+i0(d_idx)+IntIdxDisease::status).data();

        auto counter_ptr           = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::disease_counter).data();
        auto incubation_period_ptr = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::incubation_period).data();
        auto infectious_period_ptr = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::infectious_period).data();
        auto symptomdev_period_ptr = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::symptomdev_period).data();

        //auto unit_arr = pc.m_unit_mf[mfi].array();
        auto comm_arr = comm_mf[mfi].array();
        auto bx = mfi.tilebox();

        const auto* lparm = pc.getDiseaseParameters_d(d_idx);

        Gpu::DeviceScalar<int> num_infected_d(num_infected);
        int* num_infected_p = num_infected_d.dataPtr();
        ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            int community = comm_arr(i, j, k);
            if (community != random_comm) { return; }

            Box tbx;
            int i_cell = getTileIndex({AMREX_D_DECL(i, j, k)}, box, true, bin_size, tbx);
            auto cell_start = offsets[i_cell];
            auto cell_stop  = offsets[i_cell+1];
            int num_this_community = cell_stop - cell_start;
            AMREX_ASSERT(num_this_community > 0);
            //AMREX_ASSERT(cell_stop < np);

            if (num_this_community == 0) { return;}

            int ntry = 0;
            int ni = 0;
            /*unsigned*/ int stop = std::min(cell_start + ninfect, cell_stop);
            for (/*unsigned*/ int ip = cell_start; ip < stop; ++ip) {
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
                    counter_ptr[pindex] = 0;
                    incubation_period_ptr[pindex] = RandomNormal(lparm->latent_length_mean, lparm->latent_length_std, engine);
                    infectious_period_ptr[pindex] = RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                    symptomdev_period_ptr[pindex] = RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                    ++ni;
                }
            }
            *num_infected_p = ni;
        });

        Gpu::Device::streamSynchronize();
        num_infected += num_infected_d.dataValue();
        if (num_infected >= ninfect) {
            break;
        }
    }

    ParallelDescriptor::ReduceIntSum(num_infected);
    return num_infected;
}

/*! \brief Set initial cases for the simulation

    Set the initial cases of infection for the simulation based on the #CaseData:
    For each infection hub (where #CaseData::N_hubs is the number of hubs):
    + Get the FIPS code of that hub (#CaseData::FIPS_hubs)
    + Create a vector of unit numbers corresponding to that FIPS code
    + Get the number of cases for that FIPS code (#CaseData::Size_hubs)
    + Randomly infect that many agents in the units corresponding to the FIPS code, i.e.,
        cycle through units and infect agents in random communities in that unit till the
        number of infected agents is equal or greater than the number of infections for this
        FIPS code. See #ExaEpi::Initialization::infect_random_community().
*/
void setInitialCasesFromFile (AgentContainer& pc, /*!< Agent container (particle container) */
                              const std::vector<CaseData>& cases, /*!< Case data */
                              const std::vector<std::string>& d_names, /*!< Disease names */
                              const Vector<int> &FIPS_codes,
                              int ncomms,   /*!< Number of communities*/
                              const Vector<int> &unit_community_start, /*!< Start community number for each unit */
                              iMultiFab &unit_mf,
                              iMultiFab &comm_mf,
                              const bool fast_bin)
{
    BL_PROFILE("setInitialCasesFromFile");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

    for (size_t d = 0; d < cases.size(); d++) {
        amrex::Print() << "Initializing infections for " << d_names[d] << "\n";
        int ntry = 5;
        int ninf = 0;
        for (int ihub = 0; ihub < cases[d].N_hubs; ++ihub) {
            if (cases[d].Size_hubs[ihub] > 0) {
                int FIPS = cases[d].FIPS_hubs[ihub];
                std::vector<int> units;
                units.resize(0);
                for (int i = 0; i < FIPS_codes.size(); ++i) {
                    if (FIPS_codes[i] == FIPS) units.push_back(i);
                }
                //int unit = FIPS_code_to_i[FIPS];
                if (units.size() > 0) {
                    amrex::Print() << "    Attempting to infect: " << cases[d].Size_hubs[ihub] << " people in FIPS " << FIPS << "... ";
                    int u=0;
                    int i=0;
                    while (i < cases[d].Size_hubs[ihub]) {
                        int nSuccesses = infect_random_community(pc, ncomms, unit_community_start, unit_mf, comm_mf, bin_map,
                                                                 units[u], d, ntry, fast_bin);
                        ninf += nSuccesses;
                        i += nSuccesses;
                        u=(u+1)%units.size(); //sometimes we infect fewer than ntry, but switch to next unit anyway
                    }
                    amrex::Print() << "infected " << i<< " (total " << ninf << ") after processing. \n";
                }
            }
        }
        amrex::ignore_unused(ninf);
    }
}

void setInitialCasesRandom (AgentContainer& pc, /*!< Agent container (particle container) */
                            std::vector<int> num_cases, /*!< Number of initial cases */
                            const std::vector<std::string>& d_names, /*!< Disease names */
                            int ncomms,   /*!< Number of communities*/
                            const Vector<int> &unit_community_start, /*!< Start community number for each unit */
                            iMultiFab &unit_mf,
                            iMultiFab &comm_mf,
                            const bool fast_bin)
{
    BL_PROFILE("setInitialCasesRandom");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

    for (size_t d = 0; d < num_cases.size(); d++) {
        amrex::Print() << "Initializing infections for " << d_names[d] << "\n";

        int ninf = 0;
        for (int ihub = 0; ihub < num_cases[d]; ++ihub) {
            int i = 0;
            while (i < 1) {
                int nSuccesses = infect_random_community(pc, ncomms, unit_community_start, unit_mf, comm_mf, bin_map, -1, d, 1,
                                                         fast_bin);
                ninf += nSuccesses;
                i+= nSuccesses;
            }
        }
        amrex::ignore_unused(ninf);
    }
}




