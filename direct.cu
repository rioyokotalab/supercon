#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

__global__ void GPUkernel(int N, double * x, double * y, double * z, double * m,
			  double * ax, double * ay, double * az, double G, double eps) {
  int i, j, jb;
  double axi, ayi, azi, xi, yi, zi, Gmi, dx, dy, dz, R2, invR, invR3;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  axi = 0;
  ayi = 0;
  azi = 0;
  xi = x[i];
  yi = y[i];
  zi = z[i];
  Gmi = G * m[i];
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
      invR3 = invR * invR * invR * Gmi * mj[j];
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
  int N, i, j, threads;
  double OPS, G, eps, Gmi, tic, toc, diff, norm;  
  double axi, ayi, azi, dx, dy, dz, R2, invR, invR3;
  double *x, *y, *z, *m, *ax, *ay, *az;
  N = 1 << 16;
  threads = 512;
  OPS = 20. * N * N * 1e-9;
  G = 6.6743e-11;
  eps = 1e-4;
  cudaMallocManaged((void**)&x, N * sizeof(double));
  cudaMallocManaged((void**)&y, N * sizeof(double));
  cudaMallocManaged((void**)&z, N * sizeof(double));
  cudaMallocManaged((void**)&m, N * sizeof(double));
  cudaMallocManaged((void**)&ax, N * sizeof(double));
  cudaMallocManaged((void**)&ay, N * sizeof(double));
  cudaMallocManaged((void**)&az, N * sizeof(double));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N      : %d\n",N);

// CUDA
  tic = get_time();
  GPUkernel<<<N/threads,threads,threads*4*sizeof(double)>>>(N, x, y, z, m, ax, ay, az, G, eps);
  cudaThreadSynchronize();
  toc = get_time();
  printf("CUDA   : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));

// No CUDA
  diff = 0;
  norm = 0;
  tic = get_time();
#pragma omp parallel for private(axi,ayi,azi,Gmi,j,dx,dy,dz,R2,invR,invR3) reduction(+: diff, norm)
  for (i=0; i<N; i++) {
    axi = 0;
    ayi = 0;
    azi = 0;
    Gmi = G * m[i];
    for (j=0; j<N; j++) {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];
      R2 = dx * dx + dy * dy + dz * dz + eps;
      invR = 1.0f / sqrtf(R2);
      invR3 = invR * invR * invR * Gmi * m[j];
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
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(m);
  cudaFree(ax);
  cudaFree(ay);
  cudaFree(az);
  return 0;
}
