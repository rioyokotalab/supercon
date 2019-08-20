#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
  int i, N, Xmax;
  double *x, *y, *z, *m;
  double x1, x2, x3, R, X, Y, Z, S;
  FILE *file;
  N = 10000;
  x = (double*) malloc(N*sizeof(double));
  y = (double*) malloc(N*sizeof(double));
  z = (double*) malloc(N*sizeof(double));
  m = (double*) malloc(N*sizeof(double));
  i = 0;
  Xmax = 0;
  S = 3.0 * M_PI / 16.0;
  srand48(0); // random seed
  while (i < N) {
    x1 = drand48();
    x2 = drand48();
    x3 = drand48();
    R = 1.0 / sqrt( pow(x1, -2.0 / 3.0) - 1.0 );
    if (R < 100) {
      Z = (1.0 - 2.0 * x2) * R * S;
      X = sqrt(R * R - Z * Z) * cos(2.0 * M_PI * x3) * S;
      Y = sqrt(R * R - Z * Z) * sin(2.0 * M_PI * x3) * S;
      x[i] = X;
      y[i] = Y;
      z[i] = Z;
      m[i] = drand48();
      i++;
    }
  }
  file = fopen("initial.dat","wb");
  fwrite(&N,sizeof(int),1,file);
  fwrite(x,sizeof(double),N,file);
  fwrite(y,sizeof(double),N,file);
  fwrite(z,sizeof(double),N,file);
  fwrite(m,sizeof(double),N,file);
  fclose(file);
  free(x);
  free(y);
  free(z);
  free(m);
  return 0;  
}
