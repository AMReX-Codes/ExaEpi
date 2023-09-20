#include "CaseData.H"

#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>
#include <sstream>

using namespace amrex;

CaseData::CaseData (const::std::string& fname)
{
    InitFromFile(fname);
}

void CaseData::InitFromFile (const std::string& fname)
{
    BL_PROFILE("CaseData::InitFromFile");

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    FIPS_hubs.resize(0);
    Size_hubs.resize(0);
    num_cases.resize(0);
    num_cases2date.resize(0);

    FIPS_hubs_d.resize(0);
    Size_hubs_d.resize(0);
    num_cases_d.resize(0);
    num_cases2date_d.resize(0);

    num_cases.resize(57000, 0);
    num_cases2date.resize(57000, 0);

    int fips = 1;
    int last_fips = -1;
    int i, j;
    N_hubs = 0;
    int ntot = 0;
    std::string line;
    while ( (is.good()) && (fips > 0)) {
        std::getline(is, line);
        std::istringstream lis(line);
        lis >> fips >> i >> j;
        if (fips != last_fips) {
            if (fips >= 57000) {
                amrex::Abort("FIPS too large when reading case data.");
            }
            num_cases[fips] = i;
            ntot += i;
            num_cases2date[fips] = j;
            N_hubs++;
            last_fips = fips;
        } else {
            fips = -1;  // don't read another line
        }
    }

    amrex::Print() << "Setting initial case counts in " << N_hubs << " disease hubs. \n";

    FIPS_hubs.resize(N_hubs, 0);
    Size_hubs.resize(N_hubs, 0);
    j = 0;
    for (i = 0; i < 57000; i++) {
        if (num_cases[i]) {
            FIPS_hubs[j] = i;
            Size_hubs[j++] = num_cases[i];
        }
    }

    amrex::ignore_unused(ntot);
    CopyDataToDevice();
    amrex::Gpu::streamSynchronize();
}

void CaseData::Print () const {
    for (amrex::Long i = 0; i < FIPS_hubs.size(); ++i) {
        amrex::Print() << FIPS_hubs[i] << " " << num_cases[i] << " " << num_cases2date[i] << "\n";
    }
}

void CaseData::CopyToDeviceAsync (const amrex::Vector<int>& h_vec, amrex::Gpu::DeviceVector<int>& d_vec) {
    d_vec.resize(0);
    d_vec.resize(h_vec.size());
    Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
}

void CaseData::CopyToHostAsync (const amrex::Gpu::DeviceVector<int>& d_vec, amrex::Vector<int>& h_vec) {
    h_vec.resize(0);
    h_vec.resize(d_vec.size());
    Gpu::copyAsync(Gpu::deviceToHost, d_vec.begin(), d_vec.end(), h_vec.begin());
}

void CaseData::CopyDataToDevice () {
    CopyToDeviceAsync(FIPS_hubs, FIPS_hubs_d);
    CopyToDeviceAsync(Size_hubs, Size_hubs_d);
    CopyToDeviceAsync(num_cases, num_cases_d);
    CopyToDeviceAsync(num_cases2date, num_cases2date_d);
}
