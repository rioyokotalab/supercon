#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define THREADS 512

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

__global__ void GPUkernel(int N, double * x, double * y, double * z, double * m,
			  double * ax, double * ay, double * az, double G, double eps) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  double axi = 0;
  double ayi = 0;
  double azi = 0;
  double xi = x[i];
  double yi = y[i];
  double zi = z[i];
  __shared__ double xj[THREADS], yj[THREADS], zj[THREADS], mj[THREADS];
  for ( int jb=0; jb<N/THREADS; jb++ ) {
    __syncthreads();
    xj[threadIdx.x] = x[jb*THREADS+threadIdx.x];
    yj[threadIdx.x] = y[jb*THREADS+threadIdx.x];
    zj[threadIdx.x] = z[jb*THREADS+threadIdx.x];
    mj[threadIdx.x] = m[jb*THREADS+threadIdx.x];
    __syncthreads();
#pragma unroll
    for( int j=0; j<THREADS; j++ ) {
      double dx = xi - xj[j];
      double dy = yi - yj[j];
      double dz = zi - zj[j];
      double R2 = dx * dx + dy * dy + dz * dz + eps;
      double invR = rsqrtf(R2);
      double invR3 = invR * invR * invR * mj[j];
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
// Initialize
  int N = 1 << 16;
  int i, j;
  double OPS = 19. * N * N * 1e-9;
  double G = 6.6743e-11;
  double EPS = 1e-4;
  double tic, toc;
  double * x = (double*) malloc(N * sizeof(double));
  double * y = (double*) malloc(N * sizeof(double));
  double * z = (double*) malloc(N * sizeof(double));
  double * m = (double*) malloc(N * sizeof(double));
  double * ax = (double*) malloc(N * sizeof(double));
  double * ay = (double*) malloc(N * sizeof(double));
  double * az = (double*) malloc(N * sizeof(double));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N      : %d\n",N);

// CUDA
  tic = get_time();
  double *x_d, *y_d, *z_d, *m_d, *ax_d, *ay_d, *az_d;
  cudaMalloc((void**)&x_d, N * sizeof(double));
  cudaMalloc((void**)&y_d, N * sizeof(double));
  cudaMalloc((void**)&z_d, N * sizeof(double));
  cudaMalloc((void**)&m_d, N * sizeof(double));
  cudaMalloc((void**)&ax_d, N * sizeof(double));
  cudaMalloc((void**)&ay_d, N * sizeof(double));
  cudaMalloc((void**)&az_d, N * sizeof(double));
  toc = get_time();
  //printf("malloc : %e s\n",toc-tic);
  tic = get_time();
  cudaMemcpy(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(m_d, m, N * sizeof(double), cudaMemcpyHostToDevice);
  toc = get_time();
  //printf("memcpy : %e s\n",toc-tic);
  tic = get_time();
  GPUkernel<<<N/THREADS,THREADS>>>(N, x_d, y_d, z_d, m_d, ax_d, ay_d, az_d, G, EPS);
  cudaThreadSynchronize();
  toc = get_time();
  printf("CUDA   : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  tic = get_time();
  cudaMemcpy(ax, ax_d, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ay, ay_d, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(az, az_d, N * sizeof(double), cudaMemcpyDeviceToHost);
  toc = get_time();
  //printf("memcpy : %e s\n",toc-tic);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(m_d);
  cudaFree(ax_d);
  cudaFree(ay_d);
  cudaFree(az_d);

// No CUDA
  double diff = 0, norm = 0;
  tic = get_time();
#pragma omp parallel for private(j) reduction(+: diff, norm)
  for (i=0; i<N; i++) {
    double axi = 0;
    double ayi = 0;
    double azi = 0;
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      double dz = z[i] - z[j];
      double R2 = dx * dx + dy * dy + dz * dz + EPS;
      double invR = 1.0f / sqrtf(R2);
      double invR3 = invR * invR * invR * m[j];
      axi -= dx * invR3;
      ayi -= dy * invR3;
      azi -= dz * invR3;
    }
    diff += (ax[i] - axi) * (ax[i] - axi)
      + (ay[i] - ayi) * (ay[i] - ayi)
      + (az[i] - azi) * (az[i] - azi);
    norm += axi * axi + ayi * ayi + azi * azi;    
  }
  toc = get_time();
  printf("No CUDA: %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  printf("Error  : %e\n",sqrt(diff/norm));

// DEALLOCATE
  free(x);
  free(y);
  free(z);
  free(m);
  free(ax);
  free(ay);
  free(az);
  return 0;
}
