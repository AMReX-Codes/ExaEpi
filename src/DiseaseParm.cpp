#include "DiseaseParm.H"

void DiseaseParm::Initialize ()
{
    xmit_comm[0] = .0000125*pCO;
    xmit_comm[1] = .0000375*pCO;
    xmit_comm[2] = .00010*pCO;
    xmit_comm[3] = .00010*pCO;
    xmit_comm[4] = .00015*pCO;

    xmit_work = 0.115*pWO;
    infect = 1.0;
}
