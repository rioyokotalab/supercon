#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

int main(int argc, char *argv[])
{
  int i, j, N;
  size_t ret;
  struct timeval tic, toc;
  double OPS, G, eps, Gmi, time;
  double axi, ayi, azi, dx, dy, dz, R2, invR, invR3;
  double *x, *y, *z, *m, *ax, *ay, *az;
  FILE *file;
  if ( (file = fopen("initial.dat","rb")) == NULL ) {
    fprintf(stderr, "File open error.\n");
    exit(EXIT_FAILURE);
  }
  assert( fread(&N,sizeof(int),1,file) == 1 );
  OPS = 20. * N * N * 1e-9;
  x = (double*) malloc(N*sizeof(double));
  y = (double*) malloc(N*sizeof(double));
  z = (double*) malloc(N*sizeof(double));
  m = (double*) malloc(N*sizeof(double));
  ax = (double*) malloc(N*sizeof(double));
  ay = (double*) malloc(N*sizeof(double));
  az = (double*) malloc(N*sizeof(double));
  assert( fread(x,sizeof(double),N,file) == N );
  assert( fread(y,sizeof(double),N,file) == N );
  assert( fread(z,sizeof(double),N,file) == N );
  assert( fread(m,sizeof(double),N,file) == N );
  fclose(file);
  
  if ( (file = fopen("direct.dat","rb")) == NULL ) {
    fprintf(stderr, "File open error.\n");
    exit(EXIT_FAILURE);
  }
  assert( fread(&N,sizeof(int),1,file) == 1 );
  assert( fread(ax,sizeof(double),N,file) == N );
  assert( fread(ay,sizeof(double),N,file) == N );
  assert( fread(az,sizeof(double),N,file) == N );
  fclose(file);
  for (i=0; i<N; i++) {
    if ((i % (N/100)) == 0) printf("%d %e %e %e\n",i,ax[i],ay[i],az[i]);
    ax[i] /= m[i];
    ay[i] /= m[i];
    az[i] /= m[i];
  }
  file = fopen("approx.dat","wb");
  fwrite(&N,sizeof(int),1,file);
  fwrite(ax,sizeof(double),N,file);
  fwrite(ay,sizeof(double),N,file);
  fwrite(az,sizeof(double),N,file);
  fclose(file);
  free(x);
  free(y);
  free(z);
  free(m);
  free(ax);
  free(ay);
  free(az);
  return 0;  
}
