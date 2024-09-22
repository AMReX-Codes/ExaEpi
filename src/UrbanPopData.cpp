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


bool BlockGroup::read_agents(ifstream &f, Vector<UrbanPopAgent> &agents, amrex::Vector<int>& num_workgroups,
                             const int workgroup_size, const std::map<IntVect, BlockGroup> &xy_to_block_groups,
                             const LngLatToGrid &lnglat_to_grid) {
    string buf;
    num_households = 0;
    num_employed = 0;
    num_students = 0;
    AMREX_ALWAYS_ASSERT(home_population > 0);
    int start_i = agents.size();
    agents.resize(start_i + home_population);
    num_workgroups.resize(start_i + home_population);
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
            num_workgroups[i] = it->second.work_population / workgroup_size + 1;
            AMREX_ALWAYS_ASSERT(num_workgroups[i] < 5000 && num_workgroups[i] > 0);
        } else {
            num_workgroups[i] = 0;
            if (agent.role == 0) AMREX_ALWAYS_ASSERT(agent.work_lat == agent.home_lat && agent.work_lng == agent.home_lng);
            if (agent.role == 2) num_students++;
        }
    }
    num_households = households.size();

    return true;
}

bool BlockGroup::read(istringstream &iss) {
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
    min_lat = 1000;
    min_lng = 1000;
    max_lat = -1000;
    max_lng = -1000;
    for (auto &block_group : all_block_groups) {
        min_lng = min(block_group.lng, min_lng);
        max_lng = max(block_group.lng, max_lng);
        min_lat = min(block_group.lat, min_lat);
        max_lat = max(block_group.lat, max_lat);
    }
    Print() << "lng " << min_lng << ", " << max_lng << " lat " << min_lat << ", " << max_lat << "\n";

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
}


void UrbanPopData::initAgents (AgentContainer &pc, const ExaEpi::TestParams &params) {
    BL_PROFILE("UrbanPopData::init");

    LngLatToGrid lnglat_to_grid(min_lng, min_lat, gspacing_x, gspacing_y);

    int home_population = 0;
    int work_population = 0;
    int num_households = 0;
    int num_employed = 0;
    int num_students = 0;
    int num_communities = 0;
    ifstream f(params.urbanpop_filename + ".csv");
    if (!f) Abort("Could not open file " + params.urbanpop_filename + ".csv" + "\n");
    // for checking results against original urbanpop data
    std::ofstream agents_of("agents." + std::to_string(MyProc()) + ".csv");

    for (MFIter mfi = pc.MakeMFIter(0, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& tilebox = mfi.tilebox();

        Vector<UrbanPopAgent> agents;
        Vector<int> num_workgroups;
        // can't read the agent data from disk on the GPU
        for (int x = lbound(tilebox).x; x <= ubound(tilebox).x; x++) {
            for (int y = lbound(tilebox).y; y <= ubound(tilebox).y; y++) {
                auto xy = IntVect(x, y);
                auto it = xy_to_block_groups.find(xy);
                if (it != xy_to_block_groups.end()) {
                    auto &block_group = it->second;
                    num_communities++;
                    home_population += block_group.home_population;
                    work_population += block_group.work_population;
                    if (block_group.home_population > 0) {
                        // now read in the agents for this block group
                        block_group.read_agents(f, agents, num_workgroups, params.workgroup_size, xy_to_block_groups,
                                                lnglat_to_grid);
                        num_households += block_group.num_households;
                        num_employed += block_group.num_employed;
                        num_students += block_group.num_students;
                    }
                }
            }
        }

        if (num_communities == 0) continue;

        auto& ptile = pc.GetParticles(0)[{mfi.index(), mfi.LocalTileIndex()}];
        int myproc = ParallelDescriptor::MyProc();
        ptile.resize(agents.size());
        auto aos = &ptile.GetArrayOfStructs()[0];
        auto agents_ptr = agents.data();
        auto num_workgroups_ptr = num_workgroups.data();

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
        soa.GetIntData(IntIdx::withdrawn).assign(0);
        soa.GetIntData(IntIdx::random_travel).assign(0);

        ParallelForRNG (agents.size(), [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
            auto &p = aos[i];
            auto &agent = agents_ptr[i];
            p.id() = agent.id;
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
            nborhood_ptr[i] = 0;
            school_ptr[i] = agent.school_id;
            if (agent.role == 1) {
                workgroup_ptr[i] = Random_int(num_workgroups_ptr[i], engine) + 1;
                work_nborhood_ptr[i] = 0;
            } else {
                workgroup_ptr[i] = 0;
                work_nborhood_ptr[i] = 0;
            }
        });

        auto np = soa.numParticles();

        // convert device to host to avoid using managed memory. But since the outputs are only for debugging, this is overkill
        //Gpu::HostVector<int> age_group_h(np);
        //Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::age_group).begin(), soa.GetIntData(IntIdx::age_group).end(),
        //          age_group_h.begin());
        ParmParse pp("amrex");
        bool the_arena_is_managed = false;
        pp.query("the_arena_is_managed", the_arena_is_managed);
        if (the_arena_is_managed) {
            // For CUDA code, need a managed arena for this to work
            for (int i = 0; i < np; i++) {
                agents_of << aos[i].id() << " "
                        << age_group_ptr[i] << " "
                        << family_ptr[i] << " "
                        << home_i_ptr[i] << " "
                        << home_j_ptr[i] << " "
                        << work_i_ptr[i] << " "
                        << work_j_ptr[i] << " "
                        << nborhood_ptr[i] << " "
                        << school_ptr[i] << " "
                        << workgroup_ptr[i] << " "
                        << work_nborhood_ptr[i] << "\n";
            }
        }
    }

    AMREX_ALWAYS_ASSERT(pc.OK());
    agents_of.close();

    // Ugh. This crashes with:
    //   Assertion `dst.m_num_runtime_real == src.m_num_runtime_real' failed, file AMReX_ParticleTransformation.H", line 35
    // pc.WriteAsciiFile("amrex-agents.csv");


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
}

