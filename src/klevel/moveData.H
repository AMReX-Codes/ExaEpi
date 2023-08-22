#include <AMReX_Config.H>
#include <AMReX_iMultiFab.H>

using namespace amrex;

//reengineer FillPatch to support ExaEpi data
template <typename MF>
void moveDataIntraLevel (MF& mf, Real time,const Vector<MF*>& smf, const Vector<Real>& stime,int scomp, int dcomp, int ncomp,const Geometry& geom)
{
    if(smf.size()==1) {
	//the source includes only 1 patch at 1 time, we can just copy all data over
	mf.ParallelCopy(*smf[0], scomp, dcomp, ncomp, IntVect{0}, 0, geom.periodicity());
	return;
    }
    //single patch but at different times
    MF *mfptr;
    MF tempMF;
    int destcomp;
    assert(smf[0].boxArray() == smf[1].boxArray());
    assert(smf[0].DistributionMap() == smf[1].DistributionMap());
    assert(stime[0]<stime[1]);
    if (mf.boxArray() == smf[0].boxArray() && mf.DistributionMap() == smf[0].DistributionMap())
    {
	mfptr= &mf;
	destcomp= dcomp;
    }else{
        //create a temporary patch
        tempMF.define(mf->boxArray(), mf->DistributionMap(), ncomp, 0, MFInfo(), mf->Factory());
	mfptr= &tempMF;
	destcomp=0;
    }

    for(MFIter mfi(*mfptr); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const Real t0 = stime[0];
        const Real t1 = stime[1];
        auto const sfab0 = smf[0]->array(mfi);
        auto const sfab1 = smf[1]->array(mfi);
        auto       dfab  = mfptr->array(mfi);
        if (time == t0)
        {
       	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
            {
                 dfab(i,j,k,n+destcomp) = sfab0(i,j,k,n+scomp);
            });
	}
	else if (time == t1)
        {
       	    AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
            {
                 dfab(i,j,k,n+destcomp) = sfab1(i,j,k,n+scomp);
            });
	}else{//use data at the closest time, given t0<t1
	    if(time<t0 || (time-t0< t1-time)) {
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
                {
                    dfab(i,j,k,n+destcomp) = sfab0(i,j,k,n+scomp);
                });		    
	    }else{
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D ( bx, ncomp, i, j, k, n,
                {
                    dfab(i,j,k,n+destcomp) = sfab1(i,j,k,n+scomp);
                });		    
	    }
	}
    }   

    mf.ParallelCopy(*mfptr, 0, dcomp, ncomp, 0, 0, geom.periodicity());
}

#if 0
template <typename MF>EnableIf_t<IsFabArray<MF>::value>
moveDataTwoLevels (MF& mf, IntVect const& nghost, Real time,
                    const Vector<MF*>& cmf, const Vector<Real>& ct,
                    const Vector<MF*>& fmf, const Vector<Real>& ft,
                    int scomp, int dcomp, int ncomp,
                    const Geometry& cgeom, const Geometry& fgeom,
                    const IntVect& ratio)
}
#endif


