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
  
  G = 6.6743e-11;
  eps = 1e-8;
  gettimeofday(&tic,NULL);
#pragma omp parallel for private(axi,ayi,azi,G,j,dx,dy,dz,R2,invR,invR3)
  for (i=0; i<N; i++) {
    axi = 0;
    ayi = 0;
    azi = 0;
    for (j=0; j<N; j++) {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];
      R2 = dx * dx + dy * dy + dz * dz + eps;
      invR = 1 / sqrt(R2);
      invR3 = invR * invR * invR * G * m[j];
      axi -= dx * invR3;
      ayi -= dy * invR3;
      azi -= dz * invR3;
    }
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
  gettimeofday(&toc,NULL);
  time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("CPU    : %e s : %lf GFlops\n",time, OPS/time);
  file = fopen("direct.dat","wb");
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
