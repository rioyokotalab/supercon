#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

int main(int argc, char *argv[])
{
  int i, N_approx, N_direct;
  double diff, norm;
  double *ax_approx, *ay_approx, *az_approx, *ax_direct, *ay_direct, *az_direct;
  FILE *file;
  if ( (file = fopen("approx.dat","rb")) == NULL ) {
    fprintf(stderr, "File open error.\n");
    exit(EXIT_FAILURE);
  }
  assert( fread(&N_approx,sizeof(int),1,file) == 1 );
  ax_approx = (double*) malloc(N_approx*sizeof(double));
  ay_approx = (double*) malloc(N_approx*sizeof(double));
  az_approx = (double*) malloc(N_approx*sizeof(double));
  assert( fread(ax_approx,sizeof(double),N_approx,file) == N_approx );
  assert( fread(ay_approx,sizeof(double),N_approx,file) == N_approx );
  assert( fread(az_approx,sizeof(double),N_approx,file) == N_approx );

  if ( (file = fopen("direct.dat","rb")) == NULL ) {
    fprintf(stderr, "File open error.\n");
    exit(EXIT_FAILURE);
  }
  assert( fread(&N_direct,sizeof(int),1,file) == 1 );
  assert( N_approx == N_direct );
  ax_direct = (double*) malloc(N_direct*sizeof(double));
  ay_direct = (double*) malloc(N_direct*sizeof(double));
  az_direct = (double*) malloc(N_direct*sizeof(double));
  assert( fread(ax_direct,sizeof(double),N_direct,file) == N_direct );
  assert( fread(ay_direct,sizeof(double),N_direct,file) == N_direct );
  assert( fread(az_direct,sizeof(double),N_direct,file) == N_direct );
  diff = 0;
  norm = 0;
#pragma omp parallel for reduction(+: diff, norm)
  for (i=0; i<N_approx; i++) {
    diff += 
	(ax_approx[i] - ax_direct[i]) * (ax_approx[i] - ax_direct[i]) +
        (ay_approx[i] - ay_direct[i]) * (ay_approx[i] - ay_direct[i]) +
        (az_approx[i] - az_direct[i]) * (az_approx[i] - az_direct[i]);
    norm += ax_direct[i] * ax_direct[i] + ay_direct[i] * ay_direct[i] + az_direct[i] * az_direct[i];
  }
  printf("Error  : %e\n",sqrt(diff/norm));

// DEALLOCATE
  free(ax_approx);
  free(ay_approx);
  free(az_approx);
  free(ax_direct);
  free(ay_direct);
  free(az_direct);
  return 0;
}
