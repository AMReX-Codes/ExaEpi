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


struct XYLoc {
    int x,y;

    XYLoc(int x, int y) : x(x), y(y) {}

    bool operator==(const XYLoc &other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    template <>
    struct hash<XYLoc> {
        size_t operator()(const XYLoc &elem) const {
            return std::hash<int64_t>{}(elem.x) ^ (std::hash<int64_t>{}(elem.y) << 1);
        }
    };
}


bool BlockGroup::read_agents(ifstream &f, float min_lng, float min_lat, float gspacing_x, float gspacing_y) {
    /*
    string buf;
    num_employed = 0;
    num_military = 0;
    // used for counting up the number of unique households
    unordered_set<int> households;
    f.seekg(file_offset);
    // skip the first line - contains the header
    if (file_offset == 0) getline(f, buf);
    UrbanPopAgent agent;
    for (int i = 0; i < people.size(); i++) {
        if (!agent.read_csv(f))
            Abort("File is corrupted: end of file before read for offset " + to_string(file_offset) + " geoid " +
                  to_string(geoid) + "\n");
        if (agent.p_id == -1) Abort("File is corrupted: couldn't read agent p_id at offset " + to_string(file_offset) + "\n");
        if (agent.h_geoid != geoid)
            Abort("File is corrupted: wrong geoid, read " + to_string(agent.h_geoid) + " expected " + to_string(geoid) + "\n");
        households.insert(agent.h_id);
        int work_x = -1;
        int work_y = -1;
        if (agent.pr_emp_stat == 2 || agent.pr_emp_stat == 3) {
            AMREX_ALWAYS_ASSERT(agent.w_lat != -1 && agent.w_long != -1);
            work_x = (agent.w_long - min_lng) / gspacing_x;
            work_y = (agent.w_lat - min_lat) / gspacing_y;
        }
        people[i].set(agent.h_geoid, agent.w_geoid, work_x, work_y, agent.p_id, agent.h_id, agent.pr_age, agent.pr_emp_stat,
                      agent.pr_commute);
        // crude estimate based on employment status
        if (agent.pr_emp_stat == 2) num_employed++;
        if (agent.pr_emp_stat == 3) num_military++;
    }
    num_households = households.size();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(people.size() > 0, "Number of agents must be positive");
    */
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
        agents.resize(home_population);
    } catch (const std::exception &ex) {
        std::ostringstream os;
        os << "Error reading UrbanPop input file: " << ex.what() << ", line read: " << "'" << buf << "'";
        amrex::Abort(os.str());
    }
    return true;
}

static Vector<BlockGroup> read_block_groups_file(const string &fname) {
    // read in index file and broadcast
    Vector<char> idx_file_ptr;
    ParallelDescriptor::ReadAndBcastFile(fname, idx_file_ptr);
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

void UrbanPopData::construct_geom (const std::string &fname, Geometry &geom, BoxArray &ba, DistributionMapping &dm) {
    // every rank reads all the block groups from the index file
    auto all_block_groups = read_block_groups_file(fname);
    min_lat = 1000;
    min_lng = 1000;
    float max_lat = -1000;
    float max_lng = -1000;
    block_group_workers.clear();
    for (auto &block_group : all_block_groups) {
        min_lng = min(block_group.lng, min_lng);
        max_lng = max(block_group.lng, max_lng);
        min_lat = min(block_group.lat, min_lat);
        max_lat = max(block_group.lat, max_lat);
        block_group_workers[block_group.geoid] = block_group.work_population;
    }
    Print() << "lng " << min_lng << ", " << max_lng << " lat " << min_lat << ", " << max_lat << "\n";

    // grid spacing is 1/10th minute of arc at the equator, which is about 0.12 regular miles
    float gspacing = 0.1 / 60.0;
    // add a margin
    min_lng -= gspacing;
    max_lng += gspacing;
    min_lat -= gspacing;
    max_lat += gspacing;

    // the boundaries of the problem in real coordinates, i.e. latituted and longitude.
    RealBox rbox({AMREX_D_DECL(min_lng, min_lat, 0)}, {AMREX_D_DECL(max_lng, max_lat, 0)});
    // the number of grid points in a direction
    int grid_x = (max_lng - min_lng) / gspacing - 1;
    int grid_y = (max_lat - min_lat) / gspacing - 1;
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

    // create a box array with a single box representing the domain
    ba.define(geom.Domain());
    // split the box array by forcing the box size to be limited to a given number of grid points
    ba.maxSize(0.25 * grid_x / NProcs());
    //ba.maxSize(0.25 * grid_x / 8);
    Print() << "Number of boxes: " << ba.size() << "\n";
    // for checking that no x,y locations are duplicated
    std::unordered_set<XYLoc> xy_locs;
    // weights set according to population in each box so that they can be uniformly distributed
    Vector<Long> weights(ba.size(), 0);
    for (auto &block_group : all_block_groups) {
        // FIXME: check that the x,y calculated here are unique
        // convert lat/long coords to grid coords
        block_group.x = (block_group.lng - min_lng) / gspacing_x;
        block_group.y = (block_group.lat - min_lat) / gspacing_y;
        XYLoc xy_loc(block_group.x, block_group.y);
        auto it = xy_locs.find(xy_loc);
        if (it != xy_locs.end()) Abort("Found duplicate x,y location; need to decrease gspacing\n");
        else xy_locs.insert(xy_loc);
        int bi_loc = -1;
        for (int bi = 0; bi < ba.size(); bi++) {
            auto bx = ba[bi];
            if (bx.contains(IntVect(block_group.x, block_group.y))) {
                bi_loc = bi;
                weights[bi] += block_group.agents.size();
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

    block_groups.clear();
    for (auto &block_group : all_block_groups) {
        int bi_loc = -1;
        for (int bi = 0; bi < ba.size(); bi++) {
            auto bx = ba[bi];
            // do we own this box?
            if (bx.contains(IntVect(block_group.x, block_group.y))) {
                bi_loc = bi;
                if (dm[bi] == MyProc()) {
                    block_group.box_i = bi;
                    block_groups.push_back(block_group);
                }
                break;
            }
        }
        if (bi_loc == -1)
            AllPrint() << MyProc() << ": WARNING: could not find box for " << block_group.x << "," << block_group.y << "\n";
    }
}

static std::pair<int, double> get_all_load_balance (long num) {
    int all = num;
    ParallelDescriptor::ReduceIntSum(all);
    int max_num = num;
    ParallelDescriptor::ReduceIntMax(max_num);
    double load_balance = (double)all / (double)NProcs() / max_num;
    return {all, load_balance};
}



/*! \brief Read in UrbanPop data from given file
*/
void UrbanPopData::init (ExaEpi::TestParams &params, amrex::Geometry &geom, amrex::BoxArray &ba, amrex::DistributionMapping &dm)
{
    BL_PROFILE("UrbanPopData::InitFromFile");
    construct_geom(params.urbanpop_filename + ".idx", geom, ba, dm);
    // set up block groups
    int home_population = 0;
    int work_population = 0;
    int my_num_agents = 0;
    string fname = params.urbanpop_filename + ".csv";
    ifstream f(fname);
    if (!f) amrex::Abort("Could not open file " + fname + "\n");
    for (auto &block_group : block_groups) {
        my_num_agents += block_group.agents.size();
        //block_group.read_people(f, min_lng, min_lat, gspacing_x, gspacing_y);
        home_population += block_group.home_population;
        work_population += block_group.work_population;
    }
    //AllPrint() << "<" << MyProc() << ">: " << my_num_agents << " population in " << block_groups.size() << " block groups\n";
    int my_num_block_groups = block_groups.size();
    auto [all_num_block_groups, load_balance_block_groups] = get_all_load_balance(my_num_block_groups);
    auto [all_num_agents, load_balance_agents] = get_all_load_balance(my_num_agents);
    ParallelContext::BarrierAll();

    ParallelDescriptor::ReduceIntSum(home_population);
    ParallelDescriptor::ReduceIntSum(work_population);

    amrex::Print() << "Population:  " << all_num_agents << " (balance "
                                      << std::fixed << std::setprecision(3) << load_balance_agents << ")\n";
    amrex::Print() << "Home:    " << home_population << "\n";
    amrex::Print() << "Work:    " << work_population << "\n";
    //amrex::Print() << "Households:  " << num_households << "\n";
    amrex::Print() << "Communities: " << all_num_block_groups << " (balance "
                                      << std::fixed << std::setprecision(3) << load_balance_block_groups << ")\n";
}

