#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace std;

extern double size;

//
//	benchmarking program
//
int main( int argc, char **argv ) { 
	int navg, nabsavg=0;
	double dmin, absmin=1.0, davg, absavg=0.0;
	double rdavg,rdmin;
	int rnavg; 

	//
	//	process command line parameters
	//
	if( find_option( argc, argv, "-h" ) >= 0 ) {
		printf( "Options:\n" );
		printf( "-h to see this help\n" );
		printf( "-n <int> to set the number of particles\n" );
		printf( "-o <filename> to specify the output file name\n" );
		printf( "-s <filename> to specify a summary file name\n" );
		printf( "-no turns off all correctness checks and particle output\n");
		return 0;
	}
	
	const int n = read_int( argc, argv, "-n", 1000 );
	char *savename = read_string( argc, argv, "-o", NULL );
	char *sumname = read_string( argc, argv, "-s", NULL );
	
	//
	//	set up MPI
	//
	MPI_Status status;
	MPI_Request requestSend1 = MPI_REQUEST_NULL, requestSend2 = MPI_REQUEST_NULL;
	int n_proc, rank;
	MPI_Init( &argc, &argv );
	MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	int max_size = 0;
	//
	//	allocate generic resources
	//
	FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
	FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

	particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
	
	MPI_Datatype PARTICLE;
	MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
	MPI_Type_commit( &PARTICLE );
	set_size( n );

    const double cutoff_radius = 0.01;
    // size of each bin init
    const double tempSize = cutoff_radius * 5; 
	//amount per core
	const int bin_size_per_core = floor(size / tempSize);
    // amount of total bins
    const int bin_size = bin_size_per_core * n_proc;
	// size of bins after
	const double ybin_size = size / bin_size;

	//Definitions used for Indexing
	const int bottomEdge = bin_size_per_core;
	const int topEdge = 1;
	const int bottomGhost = bin_size_per_core + 1;
	const int topGhost = 0;


	if( rank == 0 ){
		init_particles( n, particles );
		cout << "n_proc: "<< n_proc << endl;
		cout << "Size of each bin before: " << size << endl;
		cout << "size of each bin after: " << ybin_size << endl;
		cout << "Number of bins: " << bin_size << endl;
		cout << "Number of bins per core: " << bin_size_per_core << endl;
		cout << endl;
	}
	//Broadcast all particles
	MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

	// vector of particles. Using +2 for the 2 ghost regions. The top ghost region is at index 0 and
	// the bottom ghost region is at bin_size_per_core+1
 	vector<particle_t> locals[bin_size_per_core +2];
	
	//loop through all the particles
	for(int i = 0; i < n; i++){
		const double lower_bound = (rank * (ybin_size *bin_size_per_core));
		const double upper_bound = ((rank+1) * (ybin_size *bin_size_per_core));

		if(lower_bound <= particles[i].y && particles[i].y < upper_bound){
			const int bin_num = (int)floor(particles[i].y / ybin_size) % bin_size_per_core;
			locals[bin_num+1].push_back(particles[i]);
		}

		//Store the particles into a ghost region vector
		if(lower_bound-ybin_size <= particles[i].y && particles[i].y < lower_bound){
			locals[topGhost].push_back(particles[i]);
		}
		if(upper_bound <= particles[i].y && particles[i].y < upper_bound+ybin_size){
			locals[bottomGhost].push_back(particles[i]);
		}
	}

	int local_size = 0, ghost_size = 0;
	for(int i = 1; i < bottomGhost; i++){
		local_size += locals[i].size();
	}
	ghost_size += locals[topGhost].size();
	ghost_size += locals[bottomGhost].size();

	// cout << "Rank " << rank << " local_size is: " << local_size << " and ghost_size is: " << ghost_size << endl;

	//
	//	simulate a number of time steps
	//
	double simulation_time = read_timer( );
	for( int step = 0; step < NSTEPS; step++ ) {
		navg = 0;
		dmin = 1.0;
		davg = 0.0;
		//
		//	save current step if necessary (slightly different semantics than in other codes)
		//
		if( find_option( argc, argv, "-no" ) == -1 )
			if( fsave && (step%SAVEFREQ) == 0 )
				save( fsave, n, particles );

		// if(locals[topGhost].size() > max_size || locals[bottomGhost].size() > max_size ){
		// 	max_size = max(locals[topGhost].size(), locals[bottomGhost].size());
		// }
		
		//
		//	compute all forces
		//
		for(int bin = 1; bin < bottomGhost; bin++ ){
			// for every particle in each bin
			for(int particle = 0; particle < locals[bin].size(); particle++) {
				// apply_force() with particles in neighboring bins
				// for every particle in the current and neighboring bins
				locals[bin][particle].ax = locals[bin][particle].ay = 0;
				for(int i = -1; i <= 1; i++ ){
					for(int other_particle = 0; other_particle < locals[bin + i].size(); other_particle++ ) {
						apply_force(locals[bin][particle], locals[bin + i][other_particle], &dmin, &davg, &navg);
					}
				}
			}
		}

		//
		//	move particles, except the ones in the ghost region
		//
		for(int bin = 1; bin < bottomGhost; bin++ ){
			for(int particle = 0; particle < locals[bin].size(); particle++) {
				move(locals[bin][particle]);
			}
		}

		//update particles to the correct bins now
		for(int bin = 1; bin < bottomGhost; bin++ ){
			// for every particle in each bin
			for(int particle = 0; particle < locals[bin].size(); particle++) {
				int newy = locals[bin][particle].y / tempSize;
				if(newy != bin){
					// add to new bin
					locals[newy].push_back(locals[bin][particle]);
					// remove from old bin
					locals[bin].erase(locals[bin].begin() + particle);
				}
			}
        }

		if (n_proc > 1) {
			// some definitions
			int static_vector = 800; //Just did testing and found that max was 693, so I set to 800
			int lastElement = static_vector - 1;

			const int receivingAbove = 1;
			const int receivingBelow = 0;

			int toGhostUpper = locals[topEdge].size();
			int toGhostLower = locals[bottomEdge].size();

			//Filler variable for resize
			particle_t EMPTY = {.x = 693,.y = 693};

			// resize edge regions to fixed size
			locals[topEdge].resize(static_vector, EMPTY);
			locals[bottomEdge].resize(static_vector, EMPTY);
			// resize ghost regions
			locals[topGhost].resize(static_vector, EMPTY);
			locals[bottomGhost].resize(static_vector, EMPTY);

			// write size to last element of edge regions
			locals[topEdge][lastElement].x = toGhostUpper;
			locals[bottomEdge][lastElement].x = toGhostLower;

			if (rank == 0) { // send to below and receive from below
				MPI_Isend(locals[bottomEdge].data(), static_vector, PARTICLE, rank + 1, receivingAbove, MPI_COMM_WORLD, &requestSend1);
				MPI_Irecv(locals[bottomGhost].data(), static_vector, PARTICLE, rank + 1, receivingBelow, MPI_COMM_WORLD, &requestSend2);
			} else if (rank == n_proc - 1) { // send to above and receive from above
				MPI_Irecv(locals[topGhost].data(), static_vector, PARTICLE, rank - 1, receivingAbove, MPI_COMM_WORLD, &requestSend1);
				MPI_Isend(locals[topEdge].data(), static_vector, PARTICLE, rank - 1, receivingBelow, MPI_COMM_WORLD, &requestSend2);
			} else { // send to above and below and receive from above and reblow
				MPI_Isend(locals[bottomEdge].data(),static_vector, PARTICLE, rank + 1, receivingAbove, MPI_COMM_WORLD,  &requestSend1);
				MPI_Isend(locals[topEdge].data(),static_vector, PARTICLE, rank - 1, receivingBelow, MPI_COMM_WORLD,&requestSend2);
				MPI_Irecv(locals[bottomGhost].data(),static_vector, PARTICLE, rank + 1, receivingBelow, MPI_COMM_WORLD,  &requestSend1);
				MPI_Irecv(locals[topGhost].data(),static_vector, PARTICLE, rank - 1, receivingAbove, MPI_COMM_WORLD, &requestSend2);
			}
			MPI_Wait(&requestSend1,&status);
			MPI_Wait(&requestSend2,&status);

			//Resizing Edges
			locals[topEdge].resize(toGhostUpper);
			locals[bottomEdge].resize(toGhostLower);

			locals[topGhost].resize(locals[topGhost][lastElement].x);
			locals[bottomGhost].resize(locals[bottomGhost][lastElement].x);
		}


		if( find_option( argc, argv, "-no" ) == -1 ) {
			MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

			if (rank == 0){
				//
				// Computing statistical data
				//
				if (rnavg) {
					absavg += rdavg/rnavg;
					nabsavg++;
				}
				if (rdmin < absmin) absmin = rdmin;
			}
		}
	}
	// cout << "Max size of the ghost region was: " << max_size << endl;

	simulation_time = read_timer( ) - simulation_time;
	
	// cout << "Finished calculations!" << endl;

	if (rank == 0) {	
		printf( "n = %d, simulation time = %g seconds", n, simulation_time);

		if( find_option( argc, argv, "-no" ) == -1 ) {
			if (nabsavg) 
				absavg /= nabsavg;
			// 
			//	-the minimum distance absmin between 2 particles during the run of the simulation
			//	-A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
			//	-A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
			//
			//	-The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
			//
			printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
			if (absmin < 0.4) 
				printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
			if (absavg < 0.8) 
				printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
		}
		printf("\n");	
			
		//	
		// Printing summary data
		//	
		if(fsum)
			fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
	}
	
	//
	//	release resources
	//
	if ( fsum )
		fclose( fsum );
	free( particles );
	if( fsave )
		fclose( fsave );
	
	MPI_Finalize( );
	
	return 0;
}
