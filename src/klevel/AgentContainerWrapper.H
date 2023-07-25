#ifndef AGENT_CONTAINERWRAPPER_H_
#define AGENT_CONTAINERWRAPPER_H_

#include "AgentContainer.H"
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_Particles.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Vector.H>

#include "DemographicData.H"
#include "DiseaseParm.H"
#include <AMReX_Amr.H>
#include <AMReX_Box.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_LevelBld.H>
#include <iostream>
#include <AMReX_ErrorList.H>

using namespace amrex;
using std::istream;
using std::ostream;


class AgentContainerWrapper
    : public AgentContainer
{
public:
    AgentContainerWrapper (const amrex::Geometry            & a_geom,
                    const amrex::DistributionMapping & a_dmap,
                    const amrex::BoxArray            & a_ba)
        : AgentContainer(a_geom, a_dmap, a_ba){}

    void copyAgentContainerToAMRGrids(amrex::MultiFab& statusMF, amrex::MultiFab& probMF, amrex::MultiFab& timerMF);
    void copyAMRGridsToAgentContainer(amrex::MultiFab& statusMF, amrex::MultiFab& probMF, amrex::MultiFab& timerMF);
};

enum StateType {State_Type = 0, Infect_Type, Status_Type, Timer_Type, Num_StateTypes};

class AgentLevel
    :
    public amrex::AmrLevel
{
public:

    MultiFab  S[Num_StateTypes];
    MultiFab  origin[Num_StateTypes];
    AgentLevel ();
    AgentLevel (amrex::Amr&            papa,
            int             lev,
            const amrex::Geometry& level_geom,
            const amrex::BoxArray& bl,
            const amrex::DistributionMapping& dm,
            amrex::Real            time);
    ~AgentLevel () override{}
    AgentLevel (const AgentLevel&) = delete;
    AgentLevel (AgentLevel&&) = delete;
    AgentLevel& operator= (const AgentLevel&) = delete;
    AgentLevel& operator= (AgentLevel&&) = delete;
    void buildStateVars(){
        printf("BUILDING VARS at level %d\n", level); 
        S[State_Type].define(grids,dmap,1,NUM_GROW);
        S[Infect_Type].define(grids,dmap,1,NUM_GROW);
        S[Status_Type].define(grids,dmap,1,NUM_GROW);
        S[Timer_Type].define(grids,dmap,1,NUM_GROW);
    };
    MultiFab& get_stateData(int lev, int state_type){
	AgentLevel* Alevel= (AgentLevel*)&parent->getLevel(lev);
        return Alevel->S[state_type];
    }
    
    void variableSetUp ();
    //void variableCleanUp ();
    void initData () override{
    }
    void setTimeLevel (amrex::Real time,
                       amrex::Real dt_old,
                       amrex::Real dt_new) {
        AmrLevel::setTimeLevel(time,dt_old,dt_new);
    }

    void init (amrex::AmrLevel& old) override{
        printf("Init AgentLevel from another level %p\n", &old); 
	//to be implemented (when needed)
    }

    void init (){
        printf("Setting time at level %d\n", level); 
	amrex::Real dt = parent->dtLevel(level);
	amrex::Real cur_time = getLevel(level - 1).state[State_Type].curTime();
	amrex::Real prev_time = getLevel(level - 1).state[State_Type].prevTime();
  	amrex::Real dt_old =
    	(cur_time - prev_time) / (amrex::Real)parent->MaxRefRatio(level - 1);
	setTimeLevel(cur_time, dt_old, dt);	
    }

    inline AgentLevel& getLevel (int lev)
    {
        return *dynamic_cast<AgentLevel *>(&parent->getLevel(lev));
    }

    amrex::Real advance (amrex::Real time,
                         amrex::Real dt,
                         int  iteration,
                         int  ncycle) override{
	printf("Advance with dt= %f\n", dt);
	if(level==0){
          pc->updateStatus();
          pc->interactAgents();
          pc->copyAgentContainerToAMRGrids(S[Status_Type], S[Infect_Type], S[Timer_Type]);
	  std::cout<<"Copy data from Agent Container to the base level"<<std::endl;
	}else{
	  if(iteration==1){
	      std::cout<<"Transfer select agents from the base level to the finer level"<<std::endl;
	      //update the transmision in the tagged area
	      const BoxArray& 	ba = S[Status_Type].boxArray ();
	      int numSelectedAgents= ba.numPts();
	      std::cout<<"numSelectedAgents "<<numSelectedAgents<<std::endl;
	      Real* status0= new Real[numSelectedAgents];
	      Real* infect0= new Real[numSelectedAgents];
	      Real* timer0= new Real[numSelectedAgents];
	      MultiFab& S_status_lev0 = get_stateData(0, Status_Type);
	      MultiFab& S_infect_lev0 = get_stateData(0, Infect_Type);
	      MultiFab& S_timer_lev0 = get_stateData(0, Timer_Type);

	      int idx=0;
	      //for simplicity, just take the first numSelectedAgents on the base level
              for(MFIter mfi(S_status_lev0); mfi.isValid(); ++mfi)
	      {
                  const Box& box = mfi.validbox();
                  auto statusArr = S_status_lev0.array(mfi);
                  auto infectArr = S_infect_lev0.array(mfi);
                  auto timerArr = S_timer_lev0.array(mfi);
                  amrex::ParallelFor(box,
                  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                  {
		      int i_idx=i-box.smallEnd()[0];
		      int j_idx=j-box.smallEnd()[1];
                      status0[idx+j_idx*box.length(0)+i_idx]=statusArr(i,j,k,0);
                      infect0[idx+j_idx*box.length(0)+i_idx]=infectArr(i,j,k,0);
                      timer0[idx+j_idx*box.length(0)+i_idx]=timerArr(i,j,k,0);
                  });
		  idx+= box.length(0)*box.length(1);
		  if(idx>=numSelectedAgents) break;
	      }

	      idx=0;
              for(MFIter mfi(S[Status_Type]); mfi.isValid(); ++mfi)
              {
                  const Box& box = mfi.validbox();
  	          auto status = S[Status_Type].array(mfi);
  	          auto infect = S[Infect_Type].array(mfi);
  	          auto timer = S[Timer_Type].array(mfi);
                  amrex::ParallelFor(box,
                  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                  {
		      int i_idx=i-box.smallEnd()[0];
		      int j_idx=j-box.smallEnd()[1];
		      status(i,j,k,0)=status0[idx+j_idx*box.length(0)+i_idx];
		      infect(i,j,k,0)=infect0[idx+j_idx*box.length(0)+i_idx];
		      timer(i,j,k,0)=timer0[idx+j_idx*box.length(0)+i_idx];
                  });
		  idx+= box.length(0)*box.length(1);
              }
	      delete[] status0;
	      delete[] infect0;
	      delete[] timer0;
	   }
	   //update at the fine  level
	   {
              for(MFIter mfi(S[Status_Type]); mfi.isValid(); ++mfi)
              {
  	          auto status = S[Status_Type].array(mfi);
  	          auto infection = S[Infect_Type].array(mfi);
  	          auto timer = S[Timer_Type].array(mfi);
                  const Box& box = mfi.validbox();
                  amrex::ParallelFor(box,
                  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                  {
                       infection(i,j,k,0) = 1.0;
                       if (status(i,j,k,0) == Status::never ||
                           status(i,j,k,0) == Status::susceptible ) {
                           return;
                       }
                       else if (status(i,j,k,0) == Status::infected) {
                           if (timer(i,j,k,0) == 0.0) {
                               status(i,j,k,0) = Status::immune;
                               timer(i,j,k,0) = 6*30*24; // 6 months in hours
                           } else {
                               timer(i,j,k,0) -= 1.0;
                           }
                       }
                       else if (status(i,j,k,0) == Status::immune) {
                           if (timer(i,j,k,0) == 0.0) {
                               status(i,j,k,0) = Status::susceptible;
                           } else {
                               timer(i,j,k,0) -= 1.0;
                           }
                       }
                      infection(i,j,k,0)+= status(i,j,k,0);
                  });
              }

              for(MFIter mfi(S[Status_Type]); mfi.isValid(); ++mfi)
              {
                  auto status = S[Status_Type].array(mfi);
                  auto infect = S[Infect_Type].array(mfi);
                  auto timer = S[Timer_Type].array(mfi);
                  const Box& box = mfi.validbox();
		  amrex::RandomEngine  engine;
                  amrex::ParallelFor(box,
                  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                  {
	                infect(i,j,k,0) = 1.0 - infect(i,j,k,0);
                	if ( status(i,j,k,0) == Status::never || status(i,j,k,0) == Status::susceptible ) {
                    	    if (amrex::Random(engine) < infect(i,j,k,0)) {
                         	status(i,j,k,0) = Status::infected;
                        	timer(i,j,k,0) = 5.0*24; // 5 days in hours
                        	return;
                	    }
			}
		  });
	      }
	   }
	   if(iteration==ncycle){
	      std::cout<<"Copy data from finer level to the coarser level"<<std::endl;
              //update the transmision in the tagged area
              const BoxArray&   ba = S[Status_Type].boxArray ();
              int numSelectedAgents= ba.numPts();
              std::cout<<"numSelectedAgents "<<numSelectedAgents<<std::endl;
              Real* status0= new Real[numSelectedAgents];
              Real* infect0= new Real[numSelectedAgents];
              Real* timer0= new Real[numSelectedAgents];
              MultiFab& S_status_lev0 = get_stateData(0, Status_Type);
              MultiFab& S_infect_lev0 = get_stateData(0, Infect_Type);
              MultiFab& S_timer_lev0 = get_stateData(0, Timer_Type);

              int idx=0;
              //for simplicity, just copy back data to the first numSelectedAgents on the base level
              for(MFIter mfi(S_status_lev0); mfi.isValid(); ++mfi)
              {
                  const Box& box = mfi.validbox();
                  auto statusArr = S_status_lev0.array(mfi);
                  auto infectArr = S_infect_lev0.array(mfi);
                  auto timerArr = S_timer_lev0.array(mfi);
                  amrex::ParallelFor(box,
                  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                  {
                      int i_idx=i-box.smallEnd()[0];
                      int j_idx=j-box.smallEnd()[1];
		      statusArr(i,j,k,0)=status0[idx+j_idx*box.length(0)+i_idx];
		      infectArr(i,j,k,0)=infect0[idx+j_idx*box.length(0)+i_idx];
		      timerArr(i,j,k,0)=timer0[idx+j_idx*box.length(0)+i_idx];
                  });
                  idx+= box.length(0)*box.length(1);
                  if(idx>=numSelectedAgents) break;
              }

              idx=0;
              for(MFIter mfi(S[Status_Type]); mfi.isValid(); ++mfi)
              {
                  const Box& box = mfi.validbox();
                  auto status = S[Status_Type].array(mfi);
                  auto infect = S[Infect_Type].array(mfi);
                  auto timer = S[Timer_Type].array(mfi);
                  amrex::ParallelFor(box,
                  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                  {
                      int i_idx=i-box.smallEnd()[0];
                      int j_idx=j-box.smallEnd()[1];
		      status0[idx+j_idx*box.length(0)+i_idx]=status(i,j,k,0);
		      infect0[idx+j_idx*box.length(0)+i_idx]=infect(i,j,k,0);
		      timer0[idx+j_idx*box.length(0)+i_idx]=timer(i,j,k,0);
                  });
                  idx+= box.length(0)*box.length(1);
              }
              delete[] status0;
              delete[] infect0;
              delete[] timer0;
	   }
	}
	return 0;
    }

    void computeInitialDt (int                   finest_level,
                           int                   /*subcycle*/,
                           amrex::Vector<int>&           n_cycle,
                           const amrex::Vector<amrex::IntVect>& ref_ratio,
                           amrex::Vector<amrex::Real>&          dt_level,
                           amrex::Real                  stop_time) override{
        dt_level[0] = 12.0;
        for (int i = 1; i <= parent->maxLevel(); i++)
        {
            dt_level[i] = dt_level[i-1]/ref_ratio[i-1][0];
            printf("Setting initial dt at level %d to %f \n", i,  dt_level[i]); 
        }    
    }

    void computeNewDt (int                   finest_level,
                       int                   /*subcycle*/,
                       amrex::Vector<int>&           n_cycle,
                       const amrex::Vector<amrex::IntVect>& ref_ratio,
                       amrex::Vector<amrex::Real>&          dt_min,
                       amrex::Vector<amrex::Real>&          dt_level,
                       amrex::Real                  stop_time,
                       int                   post_regrid_flag) override{
    
        dt_level[0] = 12.;
        for (int i = 1; i <= finest_level; i++)
        {   
            dt_level[i] = dt_level[i-1]/ref_ratio[i-1][0];
            printf("Setting new dt at level %d to %f \n", i,  dt_level[i]); 
        }
    }

    void post_regrid (int lbase,
                      int new_finest) override{}
    void post_restart () override{printf("Running post_restart()\n");}
    void post_init (amrex::Real stop_time) override{
	printf("Running post_init()\n");
    }
    void post_timestep (int iteration) override{
	printf("Running post_timestep()\n");
    }
    void postCoarseTimeStep (amrex::Real cumtime) override{
	printf("Running postCoarseTimeStep()\n");
	std::cout<<"Transfer data from the base level to the Agent Container"<<std::endl;
        pc->copyAMRGridsToAgentContainer(S[Status_Type], S[Infect_Type], S[Timer_Type]);
    }

    void errorEst (amrex::TagBoxArray& tags,
                   int          clearval,
                   int          tagval,
                   amrex::Real  time,
                   int          n_error_buf = 0,
                   int          ngrow = 0) override{
        BL_PROFILE("AgentLevel::errorEst()");
        printf("Error Estimate\n"); 

    	Real ltime = time;    
    	if (post_step_regrid) {
      		ltime = get_state_data(State_Type).curTime();
    	}
   	for (int j = 0; j < error_tags.size(); j++) {
        	std::unique_ptr<MultiFab> mf;
        	if (!error_tags[j].Field().empty()) {
            		mf = derive(error_tags[j].Field(), time, error_tags[j].NGrow());
        	}
        	error_tags[j](tags, mf.get(), TagBox::CLEAR, TagBox::SET, time, level, geom);
    	}
    	apply_problem_tags(tags, ltime);
    	apply_tagging_restrictions(tags, ltime);
    }

    
    void apply_problem_tags (amrex::TagBoxArray& tags, amrex::Real time){
    	amrex::ignore_unused(time);
	BL_PROFILE("AgentLevel::apply_problem_tags()");

    	MultiFab& S_new = get_new_data(State_Type);

    	int lev = level;
        amrex::ParmParse pp("event");
	Real finerArea=0.25; //default
        pp.get("capacity", finerArea);
	std::srand(static_cast<unsigned int>(std::time(nullptr))); 	
        Real lowerEnd= (float) rand()/RAND_MAX;
        Real higherEnd= lowerEnd+ finerArea/100.;
	if(higherEnd>=1){
	    //shift both ends toward 0
	    Real overflow=higherEnd-1;
	    lowerEnd=lowerEnd-overflow;
	    higherEnd=1;
	}

	#ifdef _OPENMP
	#pragma omp parallel
	#endif
    	{
        	for (MFIter mfi(tags); mfi.isValid(); ++mfi)
        	{
            	  const Box& bx = mfi.validbox();
            	  TagBox& tagfab = tags[mfi];
           	  auto tag_arr = tagfab.array();
            	  const auto state_arr = S_new[mfi].array();
            	  const Box& dom = geom.Domain();
		  const IntVect & problo = dom.smallEnd();
	          const IntVect & probhi = dom.bigEnd();
		  Real lo_loc[2] = {0.0};
		  Real hi_loc[2] = {0.0};
		  lo_loc[0] = problo[0]+(probhi[0]-problo[0])*lowerEnd;
		  hi_loc[0] = problo[0]+(probhi[0]-problo[0])*higherEnd;
		  lo_loc[1] = problo[1]+(probhi[1]-problo[1])*lowerEnd;
		  hi_loc[1] = problo[1]+(probhi[1]-problo[1])*higherEnd;

            	  amrex::ParallelFor(bx,
           	  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            	  {
			Real loc[2] = {0.0};
			if(i>lo_loc[0] && i<=hi_loc[0] && j>lo_loc[1] && j<=hi_loc[1])
	                	problem_tagging(i, j, k, tag_arr, state_arr, lev);
            	  });
        	}
    	}
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void problem_tagging(int i, int j, int k,
                     Array4<char> const& tag,
                     Array4<Real const> const& state,
                     int level) {
        tag(i,j,k) = TagBox::SET;
    }

    void apply_tagging_restrictions (amrex::TagBoxArray& tags, amrex::Real time){
        auto geomdata = geom.data();
        MultiFab& S_new = get_new_data(State_Type);
        int lev = level;
	
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        for (MFIter mfi(tags); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            TagBox& tagfab = tags[mfi];
            auto tag_arr = tagfab.array();
            const auto state_arr = S_new[mfi].array();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
	        tagging_restrictions(i, j, k, tag_arr, state_arr, lev, geomdata);
            });
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void tagging_restrictions(int i, int j, int k,
                     Array4<char> const& tag,
                     Array4<Real const> const& state,
                     int level, const GeometryData& geomdata) {
	    //no restriction
    }

    amrex::Real initialTimeStep (){
       Real init_dt  = 0.0;
       printf("Initial time step\n"); 
       return init_dt;	    	    
    };

    void read_params ()
    {
        printf("read parameters\n"); 
	//no extra parameter needed so far
    }

    static amrex::Vector<amrex::AMRErrorTag> error_tags;    
    int       NUM_GROW=0;
    //epi data
    AgentContainerWrapper* pc; 
};

#endif
