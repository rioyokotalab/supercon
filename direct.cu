#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define THREADS 512
typedef double real_t;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

__global__ void GPUkernel(int N, real_t * x, real_t * y, real_t * z, real_t * m,
			  real_t * ax, real_t * ay, real_t * az, real_t G, real_t eps) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  real_t axi = 0;
  real_t ayi = 0;
  real_t azi = 0;
  real_t xi = x[i];
  real_t yi = y[i];
  real_t zi = z[i];
  __shared__ real_t xj[THREADS], yj[THREADS], zj[THREADS], mj[THREADS];
  for ( int jb=0; jb<N/THREADS; jb++ ) {
    __syncthreads();
    xj[threadIdx.x] = x[jb*THREADS+threadIdx.x];
    yj[threadIdx.x] = y[jb*THREADS+threadIdx.x];
    zj[threadIdx.x] = z[jb*THREADS+threadIdx.x];
    mj[threadIdx.x] = m[jb*THREADS+threadIdx.x];
    __syncthreads();
#pragma unroll
    for( int j=0; j<THREADS; j++ ) {
      real_t dx = xj[j] - xi;
      real_t dy = yj[j] - yi;
      real_t dz = zj[j] - zi;
      real_t R2 = dx * dx + dy * dy + dz * dz + eps;
      real_t invR = rsqrtf(R2);
      real_t invR3 = mj[j] * invR * invR * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
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
  real_t OPS = 19. * N * N * 1e-9;
  real_t G = 6.6743e-11;
  real_t EPS = 1e-4;
  double tic, toc;
  real_t * x = (real_t*) malloc(N * sizeof(real_t));
  real_t * y = (real_t*) malloc(N * sizeof(real_t));
  real_t * z = (real_t*) malloc(N * sizeof(real_t));
  real_t * m = (real_t*) malloc(N * sizeof(real_t));
  real_t * ax = (real_t*) malloc(N * sizeof(real_t));
  real_t * ay = (real_t*) malloc(N * sizeof(real_t));
  real_t * az = (real_t*) malloc(N * sizeof(real_t));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N      : %d\n",N);

// CUDA
  tic = get_time();
  real_t *x_d, *y_d, *z_d, *m_d, *ax_d, *ay_d, *az_d;
  cudaMalloc((void**)&x_d, N * sizeof(real_t));
  cudaMalloc((void**)&y_d, N * sizeof(real_t));
  cudaMalloc((void**)&z_d, N * sizeof(real_t));
  cudaMalloc((void**)&m_d, N * sizeof(real_t));
  cudaMalloc((void**)&ax_d, N * sizeof(real_t));
  cudaMalloc((void**)&ay_d, N * sizeof(real_t));
  cudaMalloc((void**)&az_d, N * sizeof(real_t));
  toc = get_time();
  //printf("malloc : %e s\n",toc-tic);
  tic = get_time();
  cudaMemcpy(x_d, x, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(m_d, m, N * sizeof(real_t), cudaMemcpyHostToDevice);
  toc = get_time();
  //printf("memcpy : %e s\n",toc-tic);
  tic = get_time();
  GPUkernel<<<N/THREADS,THREADS>>>(N, x_d, y_d, z_d, m_d, ax_d, ay_d, az_d, G, EPS);
  cudaThreadSynchronize();
  toc = get_time();
  printf("CUDA   : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  tic = get_time();
  cudaMemcpy(ax, ax_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ay, ay_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(az, az_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
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
  real_t diff = 0, norm = 0;
  tic = get_time();
#pragma omp parallel for private(j) reduction(+: diff, norm)
  for (i=0; i<N; i++) {
    real_t axi = 0;
    real_t ayi = 0;
    real_t azi = 0;
    for (j=0; j<N; j++) {
      real_t dx = x[i] - x[j];
      real_t dy = y[i] - y[j];
      real_t dz = z[i] - z[j];
      real_t R2 = dx * dx + dy * dy + dz * dz + EPS;
      real_t invR = 1.0f / sqrtf(R2);
      real_t invR3 = invR * invR * invR * m[j];
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
