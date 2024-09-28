/*! @file UrbanPopData.cpp
    \brief Implementation of #UrbanPopData class
*/

#include <cmath>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include "AgentContainer.H"
#include "UrbanPopData.H"


using namespace amrex;

using std::string;
using std::to_string;
using std::unordered_map;
using std::unordered_set;
using std::ifstream;
using std::istringstream;
using std::ostringstream;
using std::runtime_error;

using ParallelDescriptor::MyProc;
using ParallelDescriptor::NProcs;


bool BlockGroup::read_agents(ifstream &f, Vector<UrbanPopAgent> &agents, Vector<int>& group_work_populations,
                             Vector<int>& group_home_populations, const std::map<IntVect, BlockGroup> &xy_to_block_groups,
                             const LngLatToGrid &lnglat_to_grid) {
    BL_PROFILE("BlockGroup::read_agents");
    string buf;
    num_households = 0;
    num_employed = 0;
    num_students = 0;
    AMREX_ALWAYS_ASSERT(home_population > 0);
    int start_i = agents.size();
    agents.resize(start_i + home_population);
    group_work_populations.resize(start_i + home_population);
    group_home_populations.resize(start_i + home_population);
    // used for counting up the number of unique households
    unordered_set<int> households;
    f.seekg(file_offset);
    // skip the first line - contains the header
    if (file_offset == 0) getline(f, buf);
    for (int i = start_i; i < agents.size(); i++) {
        auto &agent = agents[i];
        if (!agent.read_csv(f))
            Abort("File is corrupted: end of file before read for offset " + to_string(file_offset) + " geoid " +
                  to_string(geoid) + "\n");
        if (agent.id == -1) Abort("File is corrupted: couldn't read agent p_id at offset " + to_string(file_offset) + "\n");
        if (agent.home_geoid != geoid)
            Abort("File is corrupted: wrong geoid, read " + to_string(agent.home_geoid) + " expected " + to_string(geoid) + "\n");
        households.insert(agent.household_id);
        AMREX_ALWAYS_ASSERT(agent.work_lat != -1 && agent.work_lng != -1);
        if (agent.role == 1) {
            num_employed++;
            int work_x, work_y;
            lnglat_to_grid(agent.work_lng, agent.work_lat, work_x, work_y);
            auto it = xy_to_block_groups.find(IntVect(work_x, work_y));
            if (it == xy_to_block_groups.end()) Abort("Cannot find block group for work location");
            group_work_populations[i] = it->second.work_population;
        } else {
            group_work_populations[i] = 0;
            if (agent.role == 0) AMREX_ALWAYS_ASSERT(agent.work_lat == agent.home_lat && agent.work_lng == agent.home_lng);
            if (agent.role == 2) num_students++;
        }
        group_home_populations[i] = home_population;
    }
    num_households = households.size();

    return true;
}

bool BlockGroup::read(istringstream &iss) {
    BL_PROFILE("BlockGroup::read");
    const int NTOKS = 6;

    string buf;
    if (!getline(iss, buf)) return false;
    try {
        std::vector<string> tokens = split_string(buf, ' ');
        if (tokens.size() != NTOKS)
            throw runtime_error("Incorrect number of tokens, expected " + to_string(NTOKS) + " got " + to_string(tokens.size()));
        geoid = stol(tokens[0]);
        lat = stof(tokens[1]);
        lng = stof(tokens[2]);
        file_offset = stol(tokens[3]);
        home_population = stoi(tokens[4]);
        work_population = stoi(tokens[5]);
        AMREX_ALWAYS_ASSERT(home_population > 0 || work_population > 0);
    } catch (const std::exception &ex) {
        std::ostringstream os;
        os << "Error reading UrbanPop input file: " << ex.what() << ", line read: " << "'" << buf << "'";
        Abort(os.str());
    }
    return true;
}

static Vector<BlockGroup> read_block_groups_file(const string &fname) {
    BL_PROFILE("read_block_groups_file");
    // read in index file and broadcast
    Vector<char> idx_file_ptr;
    ParallelDescriptor::ReadAndBcastFile(fname  + ".idx", idx_file_ptr);
    string idx_file_ptr_string(idx_file_ptr.dataPtr());
    istringstream idx_file_iss(idx_file_ptr_string, istringstream::in);

    Vector<BlockGroup> block_groups;
    BlockGroup block_group;
    string buf;
    // first line should be column labels
    getline(idx_file_iss, buf);
    while (true) {
        if (!block_group.read(idx_file_iss)) break;
        block_groups.push_back(block_group);
    }
    return block_groups;
}

static std::pair<int, double> get_all_load_balance (const long num) {
    int all = num;
    ParallelDescriptor::ReduceIntSum(all);
    int max_num = num;
    ParallelDescriptor::ReduceIntMax(max_num);
    double load_balance = (double)all / (double)NProcs() / max_num;
    return {all, load_balance};
}

/*! \brief Read in UrbanPop data from given file
*/
void UrbanPopData::init (ExaEpi::TestParams &params, Geometry &geom, BoxArray &ba, DistributionMapping &dm) {
    BL_PROFILE("UrbanPopData::init");
    std::string fname = params.urbanpop_filename;
    // every rank reads all the block groups from the index file
    auto all_block_groups = read_block_groups_file(fname);
    // now sort block groups by geoid to make all FIPS units consecutively grouped
    std::sort(all_block_groups.begin(), all_block_groups.end(),
              [](const BlockGroup &bg1, const BlockGroup &bg2) {
                  return bg1.geoid < bg2.geoid;
              });

    min_lat = 1000;
    min_lng = 1000;
    max_lat = -1000;
    max_lng = -1000;
    for (int i = 0; i < all_block_groups.size(); i++) {
        auto& block_group = all_block_groups[i];
        min_lng = min(block_group.lng, min_lng);
        max_lng = max(block_group.lng, max_lng);
        min_lat = min(block_group.lat, min_lat);
        max_lat = max(block_group.lat, max_lat);
        block_group.block_i = i;
    }
    Print() << "lng " << min_lng << ", " << max_lng << " lat " << min_lat << ", " << max_lat << "\n";

    // get FIPS codes and community numbers from block group array
    int current_FIPS = -1;
    num_communities = 0;
    for (int i = 0; i < all_block_groups.size(); i++) {
        auto &block_group = all_block_groups[i];
        // FIPS is the first 5 digits of the GEOID, which is 12 digits
        int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);
        if (current_FIPS != fips) {
            FIPS_codes.push_back(fips);
            unit_community_start.push_back(num_communities);
            current_FIPS = fips;
        }
        num_communities++;
    }
    unit_community_start.push_back(num_communities);

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Found " << FIPS_codes.size() << " demographic units:\n";
        for (int i = 0; i < FIPS_codes.size(); i++) {
            Print() << "    FIPS " << FIPS_codes[i] << " " << unit_community_start[i] << "\n";
        }
    }

    // grid spacing is 1/10th minute of arc at the equator, which is about 0.12 regular miles
    Real gspacing = 0.1_prt / 60.0_prt;
    // add a margin
    min_lng -= gspacing;
    max_lng += gspacing;
    min_lat -= gspacing;
    max_lat += gspacing;

    // the boundaries of the problem in real coordinates, i.e. latituted and longitude.
    RealBox rbox({AMREX_D_DECL(min_lng, min_lat, 0)}, {AMREX_D_DECL(max_lng, max_lat, 0)});
    // the number of grid points in a direction
    int grid_x = (int)((max_lng - min_lng) / gspacing) - 1;
    int grid_y = (int)((max_lat - min_lat) / gspacing) - 1;
    Print() << "gspacing " << gspacing << " grid " << grid_x << ", " << grid_y << "\n";
    // the grid that overlays the domain, with the grid size in x and y directions
    Box base_domain(IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(grid_x, grid_y, 0)));
    // lat/long is a spherical coordinate system
    geom.define(base_domain, &rbox, CoordSys::SPHERICAL);
    // actual spacing (!= gspacing)
    gspacing_x = geom.CellSizeArray()[0];
    gspacing_y = geom.CellSizeArray()[1];
    Print() << "Geographic area: (" << min_lng << ", " << min_lat << ") " << max_lng << ", " << max_lat << ")\n";
    Print() << "Base domain: " << geom.Domain() << "\n";
    Print() << "Geometry: " << geom << "\n";
    Print() << "Actual grid spacing: " << gspacing_x << ", "  << gspacing_y << "\n";

    LngLatToGrid lnglat_to_grid(min_lng, min_lat, gspacing_x, gspacing_y);

    // create a box array with a single box representing the domain. Every process does this.
    ba.define(geom.Domain());
    // split the box array by forcing the box size to be limited to a given number of grid points
    ba.maxSize((int)((0.25 * grid_x) / NProcs()));
    //ba.maxSize(0.25 * grid_x / 8);
    Print() << "Number of boxes: " << ba.size() << "\n";

    // weights set according to population in each box so that they can be uniformly distributed
    // every process computes the same result - needed before distributing the boxes
    Vector<Long> weights(ba.size(), 0);
    for (auto &block_group : all_block_groups) {
        lnglat_to_grid(block_group.lng, block_group.lat, block_group.x, block_group.y);
        auto xy = IntVect(block_group.x, block_group.y);
        auto it = xy_to_block_groups.find(xy);
        if (it != xy_to_block_groups.end()) Abort("Found duplicate x,y location; need to decrease gspacing\n");
        else xy_to_block_groups.insert({xy, block_group});
        int bi_loc = -1;
        for (int bi = 0; bi < ba.size(); bi++) {
            auto bx = ba[bi];
            if (bx.contains(xy)) {
                bi_loc = bi;
                weights[bi] += block_group.home_population;
                break;
            }
        }
        if (bi_loc == -1)
            AllPrint() << MyProc() << ": WARNING: could not find box for " << block_group.x << "," << block_group.y << "\n";
    }
    //dm_latlong.SFCProcessorMap(ba, weights, NProcs());
    //Print() << "ba " << ba << " dm_latlong " << dm_latlong << "\n";
    // distribute the boxes in the array across the processors
    dm.define(ba);
    dm.KnapSackProcessorMap(weights, NProcs());

    FIPS_mf.define(ba, dm, 2, 0);
    comm_mf.define(ba, dm, 1, 0);
    FIPS_mf.setVal(-1);
    comm_mf.setVal(-1);
}


void UrbanPopData::initAgents (AgentContainer &pc, const ExaEpi::TestParams &params) {
    BL_PROFILE("UrbanPopData::initAgents");

    LngLatToGrid lnglat_to_grid(min_lng, min_lat, gspacing_x, gspacing_y);

    //auto& domain = pc.Geom(0).Domain();

    int home_population = 0;
    int work_population = 0;
    int num_households = 0;
    int num_employed = 0;
    int num_students = 0;
    num_communities = 0;
    ifstream f(params.urbanpop_filename + ".csv");
    if (!f) Abort("Could not open file " + params.urbanpop_filename + ".csv" + "\n");

    for (MFIter mfi = pc.MakeMFIter(0, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& tilebox = mfi.tilebox();

        auto FIPS_arr = FIPS_mf[mfi].array();
        auto comm_arr = comm_mf[mfi].array();

        int min_x = lbound(tilebox).x;
        int max_x = ubound(tilebox).x + 1;
        int min_y = lbound(tilebox).y;
        int max_y = ubound(tilebox).y + 1;

        Vector<UrbanPopAgent> agents;
        Vector<int> group_work_populations;
        Vector<int> group_home_populations;
        // can't read the agent data from disk on the GPU
        // FIXME: this approach requires managed memory on the GPU
        for (int x = min_x; x < max_x; x++) {
            for (int y = min_y; y < max_y; y++) {
                auto xy = IntVect(x, y);
                auto it = xy_to_block_groups.find(xy);
                if (it != xy_to_block_groups.end()) {
                    auto &block_group = it->second;
                    num_communities++;
                    home_population += block_group.home_population;
                    work_population += block_group.work_population;
                    if (block_group.home_population > 0) {
                        // now read in the agents for this block group
                        block_group.read_agents(f, agents, group_work_populations, group_home_populations, xy_to_block_groups,
                                                lnglat_to_grid);
                        num_households += block_group.num_households;
                        num_employed += block_group.num_employed;
                        num_students += block_group.num_students;

                        // FIPS is the first 5 digits of the GEOID, which is 12 digits
                        int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);
                        // Census tract is the 6 digits after the FIPS code
                        int64_t tract = static_cast<int64_t>((block_group.geoid - (fips * 1e7)) / 10);
                        FIPS_arr(x, y, 0, 0) = (int)fips;
                        FIPS_arr(x, y, 0, 1) = (int)tract;
                        comm_arr(x, y, 0) = block_group.block_i;
                    }
                }
            }
        }

        if (num_communities == 0) continue;

        int myproc = ParallelDescriptor::MyProc();
        auto& ptile = pc.DefineAndReturnParticleTile(0, mfi);
        ptile.resize(agents.size());
        auto aos = &ptile.GetArrayOfStructs()[0];
        auto agents_ptr = agents.data();
        auto group_work_populations_ptr = group_work_populations.data();
        auto group_home_populations_ptr = group_home_populations.data();

        auto& soa = ptile.GetStructOfArrays();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto family_ptr = soa.GetIntData(IntIdx::family).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        soa.GetIntData(IntIdx::hosp_i).assign(-1);
        soa.GetIntData(IntIdx::hosp_j).assign(-1);
        auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
        auto school_ptr = soa.GetIntData(IntIdx::school).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();
        int workgroup_size = params.workgroup_size;
        int nborhood_size = params.nborhood_size;
        soa.GetIntData(IntIdx::withdrawn).assign(0);
        soa.GetIntData(IntIdx::random_travel).assign(-1);

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;
        int n_disease = pc.m_num_diseases;
        for (int d = 0; d < n_disease; d++) {
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_timer).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::prob).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::incubation_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::infectious_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::symptomdev_period).assign(0.0_rt);
            soa.GetIntData(i_RT + i0(d) + IntIdxDisease::status).assign(0);
            soa.GetIntData(i_RT + i0(d) + IntIdxDisease::strain).assign(0);
            soa.GetIntData(i_RT + i0(d) + IntIdxDisease::symptomatic).assign(0);
        }
        auto np = soa.numParticles();
        AMREX_ALWAYS_ASSERT(np == agents.size());

        ParallelForRNG (np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
            auto &p = aos[i];
            auto &agent = agents_ptr[i];
            // agent ID in amrex must be > 0
            p.id() = agent.id + 1;
            p.cpu() = myproc;
            p.pos(0) = agent.home_lng;
            p.pos(1) = agent.home_lat;
            // Age group (under 5, 5-17, 18-29, 30-64, 65+)
            if (agent.age < 5) age_group_ptr[i] = 0;
            else if (agent.age < 17) age_group_ptr[i] = 1;
            else if (agent.age < 29) age_group_ptr[i] = 2;
            else if (agent.age < 64) age_group_ptr[i] = 3;
            else age_group_ptr[i] = 4;
            family_ptr[i] = agent.household_id;
            lnglat_to_grid(agent.home_lng, agent.home_lat, home_i_ptr[i], home_j_ptr[i]);
            AMREX_ALWAYS_ASSERT(tilebox.contains(IntVect(home_i_ptr[i], home_j_ptr[i])));
            lnglat_to_grid(agent.work_lng, agent.work_lat, work_i_ptr[i], work_j_ptr[i]);
            int max_nborhood = group_home_populations_ptr[i] / nborhood_size + 1;
            nborhood_ptr[i] = Random_int(max_nborhood, engine) + 1;
            school_ptr[i] = agent.school_id;
            if (agent.role == 1) {
                AMREX_ALWAYS_ASSERT(group_work_populations_ptr[i] > 0 && group_work_populations_ptr[i] < 100000);
                int max_workgroup = group_work_populations_ptr[i] / workgroup_size + 1;
                workgroup_ptr[i] = Random_int(max_workgroup, engine) + 1;
                AMREX_ALWAYS_ASSERT(workgroup_ptr[i] > 0 && workgroup_ptr[i] < 5000);
                int max_work_nborhood = group_work_populations_ptr[i] / nborhood_size + 1;
                work_nborhood_ptr[i] = Random_int(max_work_nborhood, engine) + 1;
                AMREX_ALWAYS_ASSERT(work_nborhood_ptr[i] > 0 && work_nborhood_ptr[i] < 5000);
            } else {
                workgroup_ptr[i] = 0;
                work_nborhood_ptr[i] = 0;
            }
        });

        // now ensure that all members of the same family have the same home nborhood
        ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) noexcept {
            // search forwards to find the last member of the family and use that agent's nborhood
            int nborhood = nborhood_ptr[i];
            for (int j = i + 1; j < np; j++) {
                if (home_i_ptr[i] != home_i_ptr[j] || home_j_ptr[i] != home_j_ptr[j]) break;
                if (family_ptr[i] != family_ptr[j]) break;
                nborhood = nborhood_ptr[j];
            }
            nborhood_ptr[i] = nborhood;
        });
    }

    AMREX_ALWAYS_ASSERT(pc.OK());

    AllPrint() << "Process " << MyProc() << ": population " << home_population << " in " << num_communities << " communities\n";
    auto [all_num_communities, load_balance_communities] = get_all_load_balance(num_communities);
    auto [all_num_agents, load_balance_agents] = get_all_load_balance(home_population);
    ParallelContext::BarrierAll();

    ParallelDescriptor::ReduceIntSum(home_population);
    ParallelDescriptor::ReduceIntSum(work_population);
    ParallelDescriptor::ReduceIntSum(num_households);
    ParallelDescriptor::ReduceIntSum(num_employed);
    ParallelDescriptor::ReduceIntSum(num_students);

    AMREX_ALWAYS_ASSERT(num_employed == work_population);

    Print() << std::fixed << std::setprecision(2)
            << "Population:  " << all_num_agents << " (balance " << load_balance_agents << ")\n"
            << "Employed:    " << num_employed << "\n"
            << "Students:    " << num_students << "\n"
            << "Households:  " << num_households << "\n"
            << "Communities: " << all_num_communities << " (balance " << load_balance_communities << ")\n";

    num_communities = all_num_communities;
}


