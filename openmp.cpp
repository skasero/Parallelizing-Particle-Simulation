/** 
 * Project 2 - OpenMP
 * Spencer Kase Rohlfing & Jonathan Lee
 * 
 * Our implementation of a toy particle simulator.
 * The interaction of particles considers the distance at which applicable forces
 * have an effect on individual particles and appropriately applies forces on 
 * particles with adjacent bins of particles.
 * 
 * This version utilizes the serial implementation and applies OpenMP pragmas in
 * order to efficiently parallelize individual portions of the toy particle simulation.
 * Sectios such as the moving of particles can be parallelized in its entirety without
 * having to worry about conficts and data races.
 * 
 * Applying forces and rebinning can mostly be parallelized except for the rebinning
 * which requires appropriate locks in order to ensure deadlocks do not occur and data
 * races are not apparent when individual threads try to read and write to the same bin
 * at the same time.
 **/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <iostream>
#include <vector>

using namespace std;

extern double size;

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    
    // from common.cpp
    const double cutoff_radius = 0.01;

    // size of each bin
    const double XBINSIZE = cutoff_radius;
    const double YBINSIZE = cutoff_radius;

    // amount of total bins
    const int XBINS = ceil(size / XBINSIZE);
    const int YBINS = ceil(size / YBINSIZE);

    // cout << "The total number of bins " << XBINS << " x " << YBINS << " is " << XBINS*YBINS << endl;

    vector<particle_t*> bins[XBINS][YBINS];


    //Create locks
    omp_lock_t mutex[XBINS][YBINS] ;
    for(int i = 0; i < XBINS; i++){
        for(int j = 0; j < YBINS; j++){
            omp_init_lock(&mutex[i][j]);
        }
    }
  

    int counter = 0;
    // pre-binning every particle
    for (int p = 0; p < n; p++) {
        for (int i = 0; i < XBINS; i++) {
            if (i*XBINSIZE <= particles[p].x && particles[p].x < (i+1)*XBINSIZE) {
                for (int j = 0; j < YBINS; j++) {
                    if (j*YBINSIZE <= particles[p].y && particles[p].y < (j+1)*YBINSIZE) {
                        bins[i][j].push_back(&particles[p]);
                        counter++;
                    }
                }
            }
        }
    }
    // cout << "Counter: " << counter << endl;
    
    //
    //  simulate a number of time steps
    //
    
    double simulation_time = read_timer();
    
#pragma omp parallel private(dmin)
{
    for( int step = 0; step < NSTEPS; step++ ){
        
            numthreads = omp_get_num_threads();
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            // if (step % 100 == 0) {
            //     cout << "Calculating step #" << step << " of " << NSTEPS << endl;
            //     int total = 0;
            //     for( int xbin = 0; xbin < XBINS; xbin++ ) 
            //         for( int ybin = 0; ybin < YBINS; ybin++ ) 
            //            total += bins[xbin][ybin].size();
            //     cout << "The total amount of particles is: " << total << endl;
            // }

            //
            // compute forces
            //
            
            // for every bin
            #pragma omp for collapse(2) reduction (+:navg) reduction(+:davg) schedule(static) 
            for( int xbin = 0; xbin < XBINS; xbin++ ){
                for( int ybin = 0; ybin < YBINS; ybin++ ) {
                    // for every particle in each bin
                    for( int particle = 0; particle < bins[xbin][ybin].size(); particle++ ) {
                        // apply_force() with particles in neighboring bins
                        // for every particle in the current and neighboring bins
                        // cout << "Working on particle from bin[" << xbin << "][" << ybin << "]" << endl;
                        bins[xbin][ybin][particle]->ax = bins[xbin][ybin][particle]->ay = 0;
                        for( int i = -1; i <= 1; i++ ){
                            for (int j = -1; j <= 1; j++){
                                // check for out of bounds
                                if (0 <= xbin + i && xbin + i < XBINS && 0 <= ybin + j && ybin + j < YBINS) {
                                    // cout << "Applying force using particle from bin[" << xbin + i << "][" << ybin + j << "]" << endl;
                                    for( int other_particle = 0; other_particle < bins[xbin + i][ybin + j].size(); other_particle++ ) {
                                        
                                        apply_force(*bins[xbin][ybin][particle], *bins[xbin + i][ybin + j][other_particle], &dmin, &davg, &navg);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //
            // move particles
            //

            #pragma omp for
            for(int i = 0; i < n; i++){
                move( particles[i]);
            }

            // #pragma omp for
            // for( int xbin = 0; xbin < XBINS; xbin++ ) {
            //     for( int ybin = 0; ybin < YBINS; ybin++ ) {
            //         for( int p = 0; p < bins[xbin][ybin].size(); p++ ) {
            //             move(*(bins[xbin][ybin][p]));
            //         }
            //     }}

            // for every bin
            #pragma omp for collapse(2) schedule(static)
            for( int xbin = 0; xbin < XBINS; xbin++ ) {
                for( int ybin = 0; ybin < YBINS; ybin++ ) {
                    // for every particle in each bin
                    for( int p = 0; p < bins[xbin][ybin].size(); p++ ) {
                        // move() each particle
                        omp_set_lock(&mutex[xbin][ybin]);
                        int newx = bins[xbin][ybin][p]->x / XBINSIZE;
                        int newy = bins[xbin][ybin][p]->y / YBINSIZE;
                        omp_unset_lock(&mutex[xbin][ybin]);

                        if(newx != xbin || newy != ybin){
                            
                            // add to new bin
                            omp_set_lock(&mutex[newx][newy]);
                            bins[newx][newy].push_back(bins[xbin][ybin][p]);
                            omp_unset_lock(&mutex[newx][newy]);
                            
                            // remove from old bin
                            omp_set_lock(&mutex[xbin][ybin]);
                            bins[xbin][ybin].erase(bins[xbin][ybin].begin() + p);
                            omp_unset_lock(&mutex[xbin][ybin]);
                        }
                    }
                }
            }
        
                
        if (find_option(argc, argv, "-no") == -1) {
            //
            //  compute statistical data
            //
            #pragma omp master
            if (navg) {
                absavg += davg / navg;
                nabsavg++;
            }

            #pragma omp critical
            if (dmin < absmin) absmin = dmin;

            //
            //  save if necessary
            //
            #pragma omp master
            if (fsave && (step % SAVEFREQ) == 0)
                save(fsave, n, particles);
            }
        }
    }

    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -the minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");
    
    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
