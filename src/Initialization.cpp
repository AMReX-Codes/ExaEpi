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

                    /* QDG ?? */
                    work_nborhood_ptr[ip]=4*(amrex::Random_int(4, engine))+nborhood_ptr[ip];

                    if (number) {
                        workgroup_ptr[ip] = 1 + amrex::Random_int(number, engine);
                    }
                }
            });
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
                                  const CaseData& /*cases*/, /*!< Case data */
                                  const DemographicData& demo, /*!< Demographic data */
                                  int unit, /*!< Unit number to infect */
                                  int ninfect /*!< Target number of agents to infect */ ) {
        // chose random community in unit
        int ncomms = demo.Start[unit+1] - demo.Start[unit];
        int random_comm = -1;
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
                bins.build(np, pstruct_ptr, ntiles, binner);
            }
            auto inds = bins.permutationPtr();
            auto offsets = bins.offsetsPtr();

            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();
            //auto unit_arr = unit_mf[mfi].array();
            auto comm_arr = comm_mf[mfi].array();
            auto bx = mfi.tilebox();

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
    void setInitialCases( AgentContainer&         pc,       /*!< Agent container (particle container) */
                          const amrex::iMultiFab& unit_mf,  /*!< MultiFab with unit number at each grid cell */
                          const amrex::iMultiFab& FIPS_mf,  /*!< FIPS code (component 0) and
                                                                 census tract number (component 1) */
                          const amrex::iMultiFab& comm_mf,  /*!< MultiFab with community number at each grid cell */
                          const CaseData&         cases,    /*!< Case data */
                          const DemographicData& demo       /*!< demographic data */ )
    {
        BL_PROFILE("setInitialCases");

        std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

        int ntry = 5;
        int ninf = 0;
        for (int ihub = 0; ihub < cases.N_hubs; ++ihub) {
            if (cases.Size_hubs[ihub] > 0) {
                int FIPS = cases.FIPS_hubs[ihub];
                std::vector<int> units;
                units.resize(0);
                for (int i = 0; i < demo.Nunit; ++i) if(demo.FIPS[i]==FIPS)units.push_back(i);
                //int unit = FIPS_code_to_i[FIPS];
                if (units.size() > 0) {
                    printf("Infecting %d people in FIPS %d\n", cases.Size_hubs[ihub], FIPS);
                    int u=0;
                    int i=0;
                    while (i < cases.Size_hubs[ihub]) {
                        int nSuccesses= infect_random_community(pc, unit_mf, FIPS_mf, comm_mf, bin_map, cases, demo, units[u], ntry);
                        ninf += nSuccesses;
                        i+= nSuccesses;
                        u=(u+1)%units.size(); //sometimes we infect fewer than ntry, but switch to next unit anyway
                    }
                    amrex::Print() << "Infected " << i<< " total " << ninf << " after processing FIPS " << FIPS<< " \n";
                }
            }
        }
        amrex::ignore_unused(ninf);
    }

}
}
