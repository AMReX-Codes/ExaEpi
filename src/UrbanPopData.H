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
    bool read_agents(std::ifstream &f, amrex::Vector<UrbanPopAgent> &agents, amrex::Vector<int>& num_workgroups,
                     const int workgroup_size);
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
