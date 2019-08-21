#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

__global__ void GPUkernel(int N, double * x, double * y, double * z, double * m,
			  double * ax, double * ay, double * az, double G, double eps) {
  int i, j, jb;
  double axi, ayi, azi, xi, yi, zi, dx, dy, dz, R2, invR, invR3;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  axi = 0;
  ayi = 0;
  azi = 0;
  xi = x[i];
  yi = y[i];
  zi = z[i];
  extern __shared__ double xj[];
  double *yj = &xj[blockDim.x];
  double *zj = &yj[blockDim.x];
  double *mj = &zj[blockDim.x];
  for ( jb=0; jb<N/blockDim.x; jb++ ) {
    __syncthreads();
    xj[threadIdx.x] = x[jb*blockDim.x+threadIdx.x];
    yj[threadIdx.x] = y[jb*blockDim.x+threadIdx.x];
    zj[threadIdx.x] = z[jb*blockDim.x+threadIdx.x];
    mj[threadIdx.x] = m[jb*blockDim.x+threadIdx.x];
    __syncthreads();
#pragma unroll
    for( j=0; j<blockDim.x; j++ ) {
      dx = xi - xj[j];
      dy = yi - yj[j];
      dz = zi - zj[j];
      R2 = dx * dx + dy * dy + dz * dz + eps;
      invR = rsqrtf(R2);
      invR3 = invR * invR * invR * G * mj[j];
      axi -= dx * invR3;
      ayi -= dy * invR3;
      azi -= dz * invR3;
    }
  }
  ax[i] = axi;
  ay[i] = ayi;
  az[i] = azi;
}

int main() {
  int N, threads;
  struct timeval tic, toc;
  double OPS, G, eps, time;  
  double *x, *y, *z, *m, *ax, *ay, *az;
  FILE *file;
  if ( (file = fopen("initial.dat","rb")) == NULL ) {
    fprintf(stderr, "File open error.\n");
    exit(EXIT_FAILURE);
  }
  assert( fread(&N,sizeof(int),1,file) == 1 );
  OPS = 20. * N * N * 1e-9;
  cudaMallocManaged((void**)&x, N * sizeof(double));
  cudaMallocManaged((void**)&y, N * sizeof(double));
  cudaMallocManaged((void**)&z, N * sizeof(double));
  cudaMallocManaged((void**)&m, N * sizeof(double));
  cudaMallocManaged((void**)&ax, N * sizeof(double));
  cudaMallocManaged((void**)&ay, N * sizeof(double));
  cudaMallocManaged((void**)&az, N * sizeof(double));
  assert( fread(x,sizeof(double),N,file) == N );
  assert( fread(y,sizeof(double),N,file) == N );
  assert( fread(z,sizeof(double),N,file) == N );
  assert( fread(m,sizeof(double),N,file) == N );
  threads = 500;
  G = 6.6743e-11;
  eps = 1e-8;
  gettimeofday(&tic,NULL);
  GPUkernel<<<N/threads,threads,threads*4*sizeof(double)>>>(N, x, y, z, m, ax, ay, az, G, eps);
  cudaThreadSynchronize();
  gettimeofday(&toc,NULL);
  time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("GPU    : %e s : %lf GFlops\n",time, OPS/time);
  file = fopen("direct.dat","wb");
  fwrite(&N,sizeof(int),1,file);
  fwrite(ax,sizeof(double),N,file);
  fwrite(ay,sizeof(double),N,file);
  fwrite(az,sizeof(double),N,file);
  fclose(file);
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(m);
  cudaFree(ax);
  cudaFree(ay);
  cudaFree(az);
  return 0;
}
