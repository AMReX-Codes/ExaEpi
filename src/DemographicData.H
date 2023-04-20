#ifndef DEMOGRAPHICDATA_H_
#define DEMOGRAPHICDATA_H_

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <string>

struct DemographicData
{
    DemographicData () {}

    DemographicData (const::std::string& fname);

    void InitFromFile (const std::string& fname);

    void Print () const;

    void CopyToDeviceAsync (const amrex::Vector<int>& h_vec, amrex::Gpu::DeviceVector<int>& d_vec);

    void CopyToHostAsync (const amrex::Gpu::DeviceVector<int>& d_vec, amrex::Vector<int>& h_vec);

    void CopyDataToDevice ();

    int Ncommunity;     /* number of communities required */
    int Nunit;          /* number of county/state units */
    amrex::Vector<int> myID, FIPS, Tract;    /* ID arrays */
    amrex::Vector<int> Start;         /* Starting community number for each unit */
    amrex::Vector<int> Population;    /* Population of each unit */
    amrex::Vector<int> N5, N17, N29, N64, N65plus;  /* Age-group populations */
    amrex::Vector<int> H1, H2, H3, H4, H5, H6, H7;  /* Household sizes */
    amrex::Vector<int> Ndaywork;      /* Number of daytime workers */
    amrex::Vector<int> myIDtoUnit; /* Given myID #, what Unit # is it? */
    amrex::Vector<int> Unit_on_proc; /* Is any part of this unit on this processor? */

    amrex::Gpu::DeviceVector<int> myID_d, FIPS_d, Tract_d;    /* ID arrays */
    amrex::Gpu::DeviceVector<int> Start_d;         /* Starting community number for each unit */
    amrex::Gpu::DeviceVector<int> Population_d;    /* Population of each unit */
    amrex::Gpu::DeviceVector<int> N5_d, N17_d, N29_d, N64_d, N65plus_d;  /* Age-group populations */
    amrex::Gpu::DeviceVector<int> H1_d, H2_d, H3_d, H4_d, H5_d, H6_d, H7_d;  /* Household sizes */
    amrex::Gpu::DeviceVector<int> Ndaywork_d;      /* Number of daytime workers */
    amrex::Gpu::DeviceVector<int> myIDtoUnit_d; /* Given myID #, what Unit # is it? */
    amrex::Gpu::DeviceVector<int> Unit_on_proc_d; /* Is any part of this unit on this processor? */
};

#endif
