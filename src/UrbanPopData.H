/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <string>
#include <iostream>

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include "Utils.H"
#include "CaseData.H"
#include "UrbanPopAgentStruct.H"


struct LngLatToGrid {
    Real min_lng, min_lat;
    Real gspacing_x, gspacing_y;

    LngLatToGrid(Real _min_lng, Real _min_lat, Real _gspacing_x, Real _gspacing_y) :
        min_lng(_min_lng), min_lat(_min_lat),
        gspacing_x(_gspacing_x), gspacing_y(_gspacing_y) {}

    AMREX_GPU_HOST_DEVICE
    void operator() (Real lng, Real lat, int &x, int &y) const {
        x = (int)((lng - min_lng) / gspacing_x);
        y = (int)((lat - min_lat) / gspacing_y);
    }
};


struct BlockGroup {
    int64_t geoid;
    amrex::Real lng;
    amrex::Real lat;
    size_t file_offset;
    int box_i;
    int x;
    int y;
    int home_population;
    int work_population;
    int num_households;
    int num_employed;
    int num_students;

    bool read(std::istringstream &iss);
    bool read_agents(std::ifstream &f, amrex::Vector<UrbanPopAgent> &agents, amrex::Vector<int>& group_work_population,
                     amrex::Vector<int>& group_home_population, const std::map<IntVect, BlockGroup> &xy_to_block_groups,
                     const LngLatToGrid &lnglat_to_grid);
};

struct UrbanPopData {
    amrex::Real min_lng, min_lat, max_lng, max_lat;
    amrex::Real gspacing_x, gspacing_y;

    UrbanPopData () {}

    void init(ExaEpi::TestParams &params, amrex::Geometry &geom, amrex::BoxArray &ba, amrex::DistributionMapping &dm);

    void initAgents(AgentContainer &pc, const ExaEpi::TestParams &params);

  private:

    std::map<IntVect, BlockGroup> xy_to_block_groups;

};
