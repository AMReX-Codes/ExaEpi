/*! @file Initialization.cpp
    \brief Contains initialization functions in #ExaEpi::Initialization namespace
*/

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

    /*! \brief Read worker flow data from file and set work location for agents

     *  Read in worker flow (home and work) data from a given binary file:
     *  + Initialize and allocate space for the worker-flow matrix with #DemographicData::Nunit rows
     *    and columns; note that only those rows are allocated where part of the unit resides on this
     *    processor.
     *  + Read worker flow data from #ExaEpi::TestParams::workerflow_filename: it is a binary file that
     *    contains 3 x (number of work patthers) unsigned integer data. The 3 integers are: from, to,
     *    and the number of workers with this from and to. The from and to are the IDs from the
     *    first column of the census data file (#DemographicData::myID).
     *  + For each work pattern: Read in the from, to, and number. If both the from and to ID values
     *    correspond to units that are on this processor, say, i and j, then set the worker-flow
     *    matrix element at [i][j] to the number. Note that DemographicData::myIDtoUnit() maps from
     *    ID value to unit number (from -> i, to -> j).
     *  + Comvert values in each row to row-wise cumulative values.
     *  + Scale these values to account for ~2% of people of vacation/sick leave.
     *  + For each agent (particle) in each box/tile on each processor:
     *    + Get the home (from) unit of the agent from its home cell index (i,j) and the input argument
     *      unit_mf.
     *    + Compute the number of workers in the "from" unit as 58.6% of the total population. If this
     *      number if greater than zero, continue with the following steps.
     *    + Find age group of this agent, and if it is either 18-29 or 30-64, continue with the
     *      following steps.
     *    + Assign a random work destination unit by picking a random number and placing it in the
     *      row-wise cumulative numbers in the "from" row of the worker flow matrix.
     *    + If the "to" unit is same as the "from" unit, then set the work community number same as
     *      the home community number witn 25% probability and some other random community number in
     *      the same unit with 75% probability.
     *    + Set the work location indices (i,j) for the agent to the values corresponding to this
     *      computed work community.
     *    + Find the number of workgroups in the work location unit, where one workgroup consists of
     *      20 workers; then assign a random workgroup to this agent.
    */
    void read_workerflow (const DemographicData& demo,  /*!< Demographic data */
                          const TestParams& params,     /*!< Test parameters */
                          const iMultiFab& unit_mf,     /*!< MultiFab with unit number at each grid cell */
                          const iMultiFab& comm_mf,     /*!< MultiFab with community number at each grid cell */
                          AgentContainer& pc            /*!< Agent container (particle container) */ )
    {

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
            if (from > 65334) {continue;}
            int i = demo.myIDtoUnit[from];
            if (demo.Unit_on_proc[i]) {
                if (to > 65334) {continue;}
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
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
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

                    if (number) {
                        workgroup_ptr[ip] = 1 + amrex::Random_int(number, engine);
                    }
                }
            });
        }
        assignTeachersAndWorkgroup(demo,unit_mf,comm_mf,pc);
    }

    void assignTeachersAndWorkgroup (const DemographicData& demo,  /*!< Demographic data */
                          const iMultiFab& unit_mf,     /*!< MultiFab with unit number at each grid cell */
                          const iMultiFab& comm_mf,     /*!< MultiFab with community number at each grid cell */
                          AgentContainer& pc            /*!< Agent container (particle container) */ )
    {
        const Box& domain = pc.Geom(0).Domain();

        auto total_teacher_unit = pc.getUnitTeacherCounts();

        auto total_teacher_counts = pc.getCommTeacherCounts();
        amrex::Gpu::DeviceVector<int> total_teacher_counts_mod(total_teacher_counts.size(),0);
        auto total_teacher_counts_ptr = total_teacher_counts.data();

        auto daycr_teacher_counts = pc.getCommDayCrTeacherCounts();
        amrex::Gpu::DeviceVector<int> daycr_teacher_counts_mod(daycr_teacher_counts.size(),0);
        auto daycr_teacher_counts_ptr = daycr_teacher_counts_mod.data();

        auto high_teacher_counts = pc.getCommHighTeacherCounts();
        amrex::Gpu::DeviceVector<int> high_teacher_counts_mod(high_teacher_counts.size(),0);
        auto high_teacher_counts_ptr = high_teacher_counts_mod.data();

        auto middle_teacher_counts = pc.getCommMiddleTeacherCounts();
        amrex::Gpu::DeviceVector<int> middle_teacher_counts_mod(middle_teacher_counts.size(),0);
        auto middle_teacher_counts_ptr = middle_teacher_counts_mod.data();

        auto elem3_teacher_counts = pc.getCommElem3TeacherCounts();
        amrex::Gpu::DeviceVector<int> elem3_teacher_counts_mod(elem3_teacher_counts.size(),0);
        auto elem3_teacher_counts_ptr = elem3_teacher_counts_mod.data();

        auto elem4_teacher_counts = pc.getCommElem4TeacherCounts();
        amrex::Gpu::DeviceVector<int> elem4_teacher_counts_mod(elem4_teacher_counts.size(),0);
        auto elem4_teacher_counts_ptr = elem4_teacher_counts_mod.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
            auto& soa = agents_tile.GetStructOfArrays();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
            auto school_ptr = soa.GetIntData(IntIdx::school).data();

            auto unit_arr = unit_mf[mfi].array();
            auto comm_arr = comm_mf[mfi].array();
            auto Ndaywork = demo.Ndaywork_d.data();
            auto Start = demo.Start_d.data();
            auto Ncommunity = demo.Ncommunity;

            auto np = soa.numParticles();
            for (int ip = 0; ip < np; ++ip) {

                int comm_to = (int) domain.index(IntVect(AMREX_D_DECL(work_i_ptr[ip],work_j_ptr[ip],0)));
                if (comm_to >= Ncommunity) {
                    continue;
                }
                int to = 0;
                while (comm_to >= Start[to+1]) { to++; }

                if (total_teacher_unit.data()[to] && (age_group_ptr[ip] == 2 || age_group_ptr[ip] == 3) && workgroup_ptr[ip] > 0)
                {
                    int comm_to = comm_arr(work_i_ptr[ip], work_j_ptr[ip],0);
                    int elem3_teacher  = elem3_teacher_counts_ptr[comm_to];
                    int elem4_teacher  = elem4_teacher_counts_ptr[comm_to];
                    int middle_teacher = middle_teacher_counts_ptr[comm_to];
                    int high_teacher   = high_teacher_counts_ptr[comm_to];
                    int daycr_teacher  = daycr_teacher_counts_ptr[comm_to];
                    int total          = total_teacher_counts_ptr[comm_to];

                    // 50% chance of being a teacher if in working-age population (until max_teacher_numb is met)
                    if (amrex::Random() < 0.50 && (elem3_teacher + elem4_teacher
                                                        + middle_teacher + high_teacher
                                                        + daycr_teacher) < total)
                    {
                        int available_slots[5] = {
                            elem3_teacher  < elem3_teacher_counts.data()[comm_to],
                            elem4_teacher  < elem4_teacher_counts.data()[comm_to],
                            middle_teacher < middle_teacher_counts.data()[comm_to],
                            high_teacher   < high_teacher_counts.data()[comm_to],
                            daycr_teacher  < daycr_teacher_counts.data()[comm_to]
                        };

                        int total_available = available_slots[0] + available_slots[1] + available_slots[2] + available_slots[3] + available_slots[4];
                        if (total_available > 0)
                        {
                            int choice = amrex::Random_int(total_available);
                            if (choice < available_slots[0]) {
                                school_ptr[ip] = 3;  // elementary 3 school
                                workgroup_ptr[ip] = 3 ;
                                elem3_teacher_counts_ptr[comm_to]++;
                            } else if (choice < available_slots[0] + available_slots[1]) {
                                school_ptr[ip] = 4;  // elementary 4 school
                                workgroup_ptr[ip] = 4 ;
                                elem4_teacher_counts_ptr[comm_to]++;
                            } else if (choice < available_slots[0] + available_slots[1] + available_slots[2]) {
                                school_ptr[ip] = 2;  // middle school
                                workgroup_ptr[ip] = 2 ;
                                middle_teacher_counts_ptr[comm_to]++;
                            } else if (choice < available_slots[0] + available_slots[1] + available_slots[2] + available_slots[3]) {
                                school_ptr[ip] = 1;  // high school
                                workgroup_ptr[ip] = 1 ;
                                high_teacher_counts_ptr[comm_to]++;
                            } else if (choice < total_available) {
                                school_ptr[ip] = 5;  // day care
                                workgroup_ptr[ip] = 5 ;
                                daycr_teacher_counts_ptr[comm_to]++;
                            }
                        }
                    }
                    else{
                        constexpr int WG_size = 20;
                        unsigned int number = (unsigned int) rint( ((Real) Ndaywork[to] - total_teacher_unit.data()[to] ) /
                                 ((Real) WG_size * (Start[to+1] - Start[to])) );

                        if (number) {
                            workgroup_ptr[ip] = 6 + amrex::Random_int(number);
                        }

                    }
                }

            }
        }
    }

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
    int infect_random_community ( AgentContainer& pc, /*!< Agent container (particle container)*/
                                  const amrex::iMultiFab& unit_mf, /*!< MultiFab with unit number at each grid cell */
                                  const amrex::iMultiFab& /*FIPS_mf*/, /*!< FIPS code (component 0) and
                                                                            census tract number (component 1) */
                                  const amrex::iMultiFab& comm_mf, /*!< MultiFab with community number at each grid cell */
                                  std::map<std::pair<int, int>,
                                  amrex::DenseBins<AgentContainer::ParticleType> >& bin_map, /*!< Map of dense bins with agents */
                                  const DemographicData& demo, /*!< Demographic data */
                                  int unit, /*!< Unit number to infect */
                                  const int d_idx, /*!< Disease index */
                                  int ninfect /*!< Target number of agents to infect */ ) {
        // chose random community
        int ncomms = demo.Ncommunity;
        int comm_offset = 0;
        if (unit > 0) {
            ncomms = demo.Start[unit+1] - demo.Start[unit];
            comm_offset = demo.Start[unit];
        }

        int random_comm = -1;
        if (ParallelDescriptor::IOProcessor()) {
            random_comm = amrex::Random_int(ncomms) + comm_offset;
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
            amrex::DenseBins<AgentContainer::ParticleType>& bins = bin_map[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
            auto& aos = agents_tile.GetArrayOfStructs();
            auto& soa = agents_tile.GetStructOfArrays();
            const size_t np = aos.numParticles();
            if (np == 0) { continue; }
            auto pstruct_ptr = aos().dataPtr();
            const Box& box = mfi.validbox();

            int ntiles = numTilesInBox(box, true, bin_size);

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, box};
            if (bins.numBins() < 0) {
                bins.build(BinPolicy::Serial, np, pstruct_ptr, ntiles, binner);
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

            //auto unit_arr = unit_mf[mfi].array();
            auto comm_arr = comm_mf[mfi].array();
            auto bx = mfi.tilebox();

            const auto* lparm = pc.getDiseaseParameters_d(d_idx);

            Gpu::DeviceScalar<int> num_infected_d(num_infected);
            int* num_infected_p = num_infected_d.dataPtr();
            amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
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
                        incubation_period_ptr[pindex] = amrex::RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                        infectious_period_ptr[pindex] = amrex::RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                        symptomdev_period_ptr[pindex] = amrex::RandomNormal(lparm->symptomdev_length_mean, lparm->symptomdev_length_std, engine);
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
                                  const amrex::iMultiFab& unit_mf, /*!< MultiFab with unit number at each grid cell */
                                  const amrex::iMultiFab& FIPS_mf, /*!< FIPS code (component 0) and
                                                                        census tract number (component 1) */
                                  const amrex::iMultiFab& comm_mf, /*!< MultiFab with community number at each grid cell */
                                  const std::vector<CaseData>& cases, /*!< Case data */
                                  const std::vector<std::string>& d_names, /*!< Disease names */
                                  const DemographicData& demo /*!< demographic data */ )
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
                    for (int i = 0; i < demo.Nunit; ++i) if(demo.FIPS[i]==FIPS)units.push_back(i);
                    //int unit = FIPS_code_to_i[FIPS];
                    if (units.size() > 0) {
                        amrex::Print() << "    Attempting to infect: " << cases[d].Size_hubs[ihub] << " people in FIPS " << FIPS << "... ";
                        int u=0;
                        int i=0;
                        while (i < cases[d].Size_hubs[ihub]) {
                            int nSuccesses= infect_random_community(pc, unit_mf, FIPS_mf, comm_mf, bin_map, demo, units[u], d, ntry);
                            ninf += nSuccesses;
                            i+= nSuccesses;
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
                                const amrex::iMultiFab& unit_mf, /*!< MultiFab with unit number at each grid cell */
                                const amrex::iMultiFab& FIPS_mf, /*!< FIPS code (component 0) and
                                                                      census tract number (component 1) */
                                const amrex::iMultiFab& comm_mf, /*!< MultiFab with community number at each grid cell */
                                std::vector<int> num_cases, /*!< Number of initial cases */
                                const std::vector<std::string>& d_names, /*!< Disease names */
                                const DemographicData& demo /*!< demographic data */ )
    {
        BL_PROFILE("setInitialCasesRandom");

        std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

        for (size_t d = 0; d < num_cases.size(); d++) {
            amrex::Print() << "Initializing infections for " << d_names[d] << "\n";

            int ninf = 0;
            for (int ihub = 0; ihub < num_cases[d]; ++ihub) {
                int i = 0;
                while (i < 1) {
                    int nSuccesses= infect_random_community(pc, unit_mf, FIPS_mf, comm_mf, bin_map, demo, -1, d, 1);
                    ninf += nSuccesses;
                    i+= nSuccesses;
                }
            }
            amrex::ignore_unused(ninf);
        }
    }

}
}
