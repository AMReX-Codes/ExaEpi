/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <string>
#include <iostream>

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include "Utils.H"
#include "UrbanPopAgentStruct.H"


struct BlockGroup {
    int64_t geoid;
    float lng;
    float lat;
    size_t file_offset;
    int box_i;
    int x;
    int y;
    int home_population;
    int work_population;
    amrex::Vector<UrbanPopAgent> agents;

    bool read(std::istringstream &iss);
    bool read_agents(std::ifstream &f, float min_lng, float min_lat, float gspacing_x, float gspacing_y);
};

struct UrbanPopData {
    // each block group has around 1500 people
    amrex::Vector<BlockGroup> block_groups;
    std::unordered_map<int64_t, int> block_group_workers;
    float min_lng, min_lat;
    float gspacing_x, gspacing_y;

    /*! \brief Null constructor */
    UrbanPopData () {}

    void init(ExaEpi::TestParams &params, amrex::Geometry &geom, amrex::BoxArray &ba, amrex::DistributionMapping &dm);

  private:

    void construct_geom(const std::string &fname, amrex::Geometry &geom, amrex::BoxArray &ba, amrex::DistributionMapping &dm);

};
