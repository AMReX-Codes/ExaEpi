/*! @file DiseaseParm.cpp
    \brief Function implementations for #DiseaseParm class
*/

#include "DiseaseParm.H"

#include "AMReX_Print.H"

/*! \brief Initialize disease parameters

    Compute transmission probabilities for various situations based on disease
    attributes.
*/
void DiseaseParm::Initialize ()
{
    xmit_comm[0] = .0000125*pCO;
    xmit_comm[1] = .0000375*pCO;
    xmit_comm[2] = .00010*pCO;
    xmit_comm[3] = .00010*pCO;
    xmit_comm[4] = .00015*pCO;

    xmit_hood[0] = .00005*pNH;
    xmit_hood[1] = .00015*pNH;
    xmit_hood[2] = xmit_hood[3] = .00040*pNH;
    xmit_hood[4] = .00060*pNH;

    xmit_nc_adult[0] = xmit_nc_adult[1] = .08*pHC;
    xmit_nc_adult[2] = xmit_nc_adult[3] = xmit_nc_adult[4] = .1*pHC;

    xmit_nc_child[0] = xmit_nc_child[1] = .15*pHC;
    xmit_nc_child[2] = xmit_nc_child[3] = xmit_nc_child[4] = .08*pHC;

    xmit_work = 0.115*pWO;

    // Optimistic scenario: 50% reduction in external child contacts during school dismissal
    //   or remote learning, and no change in household contacts
    Child_compliance=0.5; Child_HH_closure=1.0;
    // Pessimistic scenario: 30% reduction in external child contacts during school dismissal
    //   or remote learning, and 2x increase in household contacts
    //  sch_compliance=0.3; sch_effect=2.0;

    /*
      Double household contact rate involving children, and reduce
      other child-related contacts (neighborhood cluster, neigborhood,
      and community) by the compliance rate, Child_compliance
    */
    for (int i = 0; i < 5; i++) {
        xmit_child_SC[i] = xmit_child[i] * Child_HH_closure;
        xmit_nc_child_SC[i] = xmit_nc_child[i] * (1.0 - Child_compliance);
    }
    for (int i = 0; i < 2; i++) {
        xmit_adult_SC[i] = xmit_adult[i] * Child_HH_closure;
        xmit_nc_adult_SC[i] = xmit_nc_adult[i] * (1.0 - Child_compliance);
        xmit_comm_SC[i] = xmit_comm[i] * (1.0 - Child_compliance);
        xmit_hood_SC[i] = xmit_hood[i] * (1.0 - Child_compliance);
    }
    for (int i = 2; i < 5; i++) {
        xmit_adult_SC[i] = xmit_adult[i];
        xmit_nc_adult_SC[i] = xmit_nc_adult[i];    // Adult-only contacts remain unchanged
        xmit_comm_SC[i] = xmit_comm[i];
        xmit_hood_SC[i] = xmit_hood[i];
    }

    // Multiply contact rates by transmission probability given contact
    xmit_work *= p_trans[0];

    for (int i = 0; i < 5; i++) {
        xmit_comm[i] *= p_trans[0];
        xmit_hood[i] *= p_trans[0];
        xmit_nc_adult[i] *= p_trans[0];
        xmit_nc_child[i] *= p_trans[0];
        xmit_adult[i] *= p_trans[0];
        xmit_child[i] *= p_trans[0];
    }

    for (int i = 1; i < 7; i++) xmit_school[i] *= p_trans[0];
    for (int i = 1; i < 5; i++) {
        xmit_child_SC[i] *= p_trans[0];
        xmit_adult_SC[i] *= p_trans[0];
        xmit_nc_child_SC[i] *= p_trans[0];
        xmit_nc_adult_SC[i] *= p_trans[0];
        xmit_comm_SC[i] *= p_trans[0];
        xmit_hood_SC[i] *= p_trans[0];
        xmit_sch_c2a[i] *= p_trans[0];
        xmit_sch_a2c[i] *= p_trans[0];
    }

    infect = 1.0;
}

void DiseaseParm::printMatrix () {
    amrex::Print() << "xmit_comm: " << " ";
    for (int i = 0; i < 5; ++i) {
        amrex::Print() << xmit_comm[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_hood: " <<  " ";
    for (int i = 0; i < 5; ++i) {
        amrex::Print() << xmit_hood[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_adult: " << " ";
    for (int i = 0; i < 5; ++i) {
        amrex::Print() << xmit_nc_adult[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_child: " << " ";
    for (int i = 0; i < 5; ++i) {
        amrex::Print() << xmit_nc_child[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_work: " << " ";
    amrex::Print() << xmit_work << "\n";

    amrex::Print() << "xmit_child_SC: " << " ";
    for (int i = 0; i < 5; ++i) {
        amrex::Print() << xmit_child_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_child_SC: " << " ";
    for (int i = 0; i < 5; ++i) {
        amrex::Print() << xmit_nc_child_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_adult_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_adult_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_adult_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_nc_adult_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_hood_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_hood_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_comm_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_comm_SC[i] << " ";
    }
    amrex::Print() << "\n";
}
