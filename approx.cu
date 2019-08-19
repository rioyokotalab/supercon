#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <assert.h>

#define P 10
#define MTERM P*(P+1)*(P+2)/6
#define CTERM (P+1)*(P+2)*(P+3)/6

//! Structure of bodies
struct Body {
  int i;
  double X[3];
  double m;
  double F[3];
};

//! Structure of nodes
struct Node {
  int numChilds;
  int numBodies;
  double X[3];
  double R;
  struct Node * child;
  struct Body * body;
  double M[MTERM];
};

double timeDiff(struct timeval tic, struct timeval toc) {
  return toc.tv_sec - tic.tv_sec + (toc.tv_usec - tic.tv_usec) * 1e-6;
}

void initBodies(struct Body * bodies, int numBodies) {
  srand48(0);
  int b = 0;
  while (b < numBodies) {
    double X1 = drand48();
    double X2 = drand48();
    double X3 = drand48();
    double R = 1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );
    if (R < 100.0) {
      double Z = (1.0 - 2.0 * X2) * R;
      double X = sqrt(R * R - Z * Z) * cos(2.0 * M_PI * X3);
      double Y = sqrt(R * R - Z * Z) * sin(2.0 * M_PI * X3);
      double scale = 3.0 * M_PI / 16.0;
      X *= scale; Y *= scale; Z *= scale;
      bodies[b].X[0] = X;
      bodies[b].X[1] = Y;
      bodies[b].X[2] = Z;
      bodies[b].m = drand48();
      bodies[b].F[0] = 0;
      bodies[b].F[1] = 0;
      bodies[b].F[2] = 0;
      b++;
    }
  }
}

//! Get bounding box of bodies
void getBounds(struct Body * bodies, int numBodies, double * X0, double * R0) {
  double Xmin[3], Xmax[3];
  for (int d=0; d<3; d++) {
    Xmin[d] = bodies[0].X[d];
    Xmax[d] = bodies[0].X[d];
  }
  for (size_t b=1; b<numBodies; b++) {
    for (int d=0; d<3; d++) {
      Xmin[d] = fmin(bodies[b].X[d], Xmin[d]);
      Xmax[d] = fmax(bodies[b].X[d], Xmax[d]);
    }
  }
  *R0 = 0;
  for (int d=0; d<3; d++) {
    X0[d] = (Xmax[d] + Xmin[d]) / 2;
    *R0 = fmax(X0[d]-Xmin[d], *R0);
    *R0 = fmax(Xmax[d]-X0[d], *R0);
  }
  *R0 *= 1.00001;
}

//! Build nodes of tree adaptively using a top-down approach based on recursion
void buildTree(struct Body * bodies, struct Body * buffer, int begin, int end,
               struct Node * node, struct Node * node_p,
               int * numNodes, double * X, double R, double ncrit, bool direction) {
  //! Create a tree node
  node->body = bodies + begin;
  if(direction) node->body = buffer + begin;
  node->numBodies = end - begin;
  node->numChilds = 0;
  for (int d=0; d<3; d++) node->X[d] = X[d];
  node->R = R;
  //! Count number of bodies in each octant
  int size[8] = {0,0,0,0,0,0,0,0};
  double x[3];
  for (int i=begin; i<end; i++) {
    for (int d=0; d<3; d++) x[d] = bodies[i].X[d];
    int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
    size[octant]++;
  }
  //! Exclusive scan to get offsets
  int offset = begin;
  int offsets[8], counter[8];
  for (int i=0; i<8; i++) {
    offsets[i] = offset;
    offset += size[i];
    if (size[i]) node->numChilds++;
  }
  //! If node is a leaf
  if (end - begin <= ncrit) {
    node->numChilds = 0;
    if (direction) {
      for (int i=begin; i<end; i++) {
        for (int d=0; d<3; d++) buffer[i].X[d] = bodies[i].X[d];
        buffer[i].m = bodies[i].m;
        buffer[i].i = bodies[i].i;
      }
    }
    return;
  }
  //! Sort bodies by octant
  for (int i=0; i<8; i++) counter[i] = offsets[i];
  for (int i=begin; i<end; i++) {
    for (int d=0; d<3; d++) x[d] = bodies[i].X[d];
    int octant = (x[0] > X[0]) + ((x[1] > X[1]) << 1) + ((x[2] > X[2]) << 2);
    for (int d=0; d<3; d++) buffer[counter[octant]].X[d] = bodies[i].X[d];
    buffer[counter[octant]].m = bodies[i].m;
    buffer[counter[octant]].i = bodies[i].i;
    counter[octant]++;
  }
  //! Loop over children and recurse
  double Xchild[3];
  struct Node * child = node_p + *numNodes + 1;
  *numNodes += node->numChilds;
  node->child = child;
  for (int i=0, c=0; i<8; i++) {
    for (int d=0; d<3; d++) Xchild[d] = X[d];
    double Rchild = R / 2;
    for (int d=0; d<3; d++) {
      Xchild[d] += Rchild * (((i & 1 << d) >> d) * 2 - 1);
    }
    if (size[i]) {
      buildTree(buffer, bodies, offsets[i], offsets[i] + size[i],
                 &child[c++], node_p, numNodes, Xchild, Rchild, ncrit, !direction);
    }
  }
}

int indexP(int nx, int ny, int nz, int p) {
  int psum = p * (p+1) * (p+2) / 6;
  int pxsum = (p - nx - 1) * (p - nx) * (p - nx + 1) / 6;
  int pxysum = (p - nx - ny) * (p - nx - ny + 1) / 2;
  return psum - pxsum - pxysum + nz;
}
  
void P2P(struct Node * Ci, struct Node * Cj) {
  double eps = 1e-8;
  struct Body * Bi = Ci->body;
  struct Body * Bj = Cj->body;
  for (int i=0; i<Ci->numBodies; i++) {
    double F[3] = {0, 0, 0};
    for (int j=0; j<Cj->numBodies; j++) {
      double dX[3];
      for (int d=0; d<3; d++) dX[d] = Bi[i].X[d] - Bj[j].X[d];
      double R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2] + eps;
      double invR2 = 1.0 / R2;
      double invR = Bj[j].m * sqrt(invR2);
      for (int d=0; d<3; d++) F[d] += dX[d] * invR2 * invR;
    }
    for (int d=0; d<3; d++) {
#pragma omp atomic
      Bi[i].F[d] -= F[d];
    }
  }
}

void P2M(struct Node * C) {
  for (struct Body * B=C->body; B!=C->body+C->numBodies; B++) {
    double dX[3], Mx[P], My[P], Mz[P];
    for (int d=0; d<3; d++) dX[d] = C->X[d] - B->X[d];
    Mx[0] = My[0] = Mz[0] = 1;
    for (int n=1; n<P; n++) {
      Mx[n] = Mx[n-1] * dX[0] / n;
      My[n] = My[n-1] * dX[1] / n;
      Mz[n] = Mz[n-1] * dX[2] / n;
    }
    double M[MTERM];
    for (int nx=0; nx<P; nx++) {
      for (int ny=0; ny<P-nx; ny++) {
        for (int nz=0; nz<P-nx-ny; nz++) {
          M[indexP(nx,ny,nz,P)] = B->m * Mx[nx] * My[ny] * Mz[nz];
        }
      }
    }
    for (int i=0; i<MTERM; i++) C->M[i] += M[i];
  }
}

void M2M(struct Node * Ci) {
  for (struct Node * Cj=Ci->child; Cj!=Ci->child+Ci->numChilds; Cj++) {
    double dX[3], Mx[P], My[P], Mz[P];
    for (int d=0; d<3; d++) dX[d] = Ci->X[d] - Cj->X[d];
    Mx[0] = My[0] = Mz[0] = 1;
    for (int n=1; n<P; n++) {
      Mx[n] = Mx[n-1] * dX[0] / n;
      My[n] = My[n-1] * dX[1] / n;
      Mz[n] = Mz[n-1] * dX[2] / n;
    }
    double C[MTERM];
    for (int nx=0; nx<P; nx++) {
      for (int ny=0; ny<P-nx; ny++) {
        for (int nz=0; nz<P-nx-ny; nz++) {
          C[indexP(nx,ny,nz,P)] = Mx[nx] * My[ny] * Mz[nz];
        }
      }
    }
    double M[MTERM];
    for (int i=0; i<MTERM; i++) M[i] = Cj->M[i];
    for (int nx=0; nx<P; nx++) {
      for (int ny=0; ny<P-nx; ny++) {
        for (int nz=0; nz<P-nx-ny; nz++) {
          for (int kx=0; kx<=nx; kx++) {
            for (int ky=0; ky<=ny; ky++) {
              for (int kz=0; kz<=nz; kz++) {
                Ci->M[indexP(nx,ny,nz,P)] += C[indexP(nx-kx,ny-ky,nz-kz,P)] * M[indexP(kx,ky,kz,P)];
              }
            }
          }
        }
      }
    }
  }
}

void M2P(struct Node * Ci, struct Node * Cj) {
  for (struct Body * B=Ci->body; B!=Ci->body+Ci->numBodies; B++) {
    double dX[3];
    for (int d=0; d<3; d++) dX[d] = B->X[d] - Cj->X[d];
    double x = dX[0], y = dX[1], z = dX[2];
    double R2 = x * x + y * y + z * z;
    double invR2 = 1 / R2;
    double invR  = sqrtf(invR2);
    double invR3 = invR * invR2;
    double factorial[P+1];
    factorial[0] = 1;
    for (int n=1; n<=P; n++) {
      factorial[n] = factorial[n-1] * n;
    }
    double C[CTERM];
    C[indexP(0,0,0,P+1)] = invR;
    C[indexP(1,0,0,P+1)] = - x * invR3;
    C[indexP(0,1,0,P+1)] = - y * invR3;
    C[indexP(0,0,1,P+1)] = - z * invR3;
    for (int n=2; n<=P; n++) {
      for (int nx=0; nx<=n; nx++) {
        for (int ny=0; ny<=n-nx; ny++) {
          int nz = n-nx-ny;
          double C1x = nx < 1 ? 0 : C[indexP(nx-1,ny,nz,P+1)];
          double C1y = ny < 1 ? 0 : C[indexP(nx,ny-1,nz,P+1)];
          double C1z = nz < 1 ? 0 : C[indexP(nx,ny,nz-1,P+1)];
          double C2x = nx < 2 ? 0 : C[indexP(nx-2,ny,nz,P+1)];
          double C2y = ny < 2 ? 0 : C[indexP(nx,ny-2,nz,P+1)];
          double C2z = nz < 2 ? 0 : C[indexP(nx,ny,nz-2,P+1)];
          C[indexP(nx,ny,nz,P+1)] = ((1-2*n)*(x * C1x + y * C1y + z * C1z) +
                                    (1-n)*(C2x + C2y + C2z)) * invR2 / n;
        }
      }
    }
    for (int n=0; n<=P; n++) {
      for (int nx=0; nx<=n; nx++) {
        for (int ny=0; ny<=n-nx; ny++) {
          int nz = n-nx-ny;
          C[indexP(nx,ny,nz,P+1)] *= factorial[nx] * factorial[ny] * factorial[nz];
        }
      }
    }
    for (int nx=0; nx<P; nx++) {
      for (int ny=0; ny<P-nx; ny++) {
        for (int nz=0; nz<P-nx-ny; nz++) {
          B->F[0] += C[indexP(nx+1,ny,nz,P+1)] * Cj->M[indexP(nx,ny,nz,P)];
          B->F[1] += C[indexP(nx,ny+1,nz,P+1)] * Cj->M[indexP(nx,ny,nz,P)];
          B->F[2] += C[indexP(nx,ny,nz+1,P+1)] * Cj->M[indexP(nx,ny,nz,P)];
        }
      } 
    }
  }
}

//! Recursive call to post-order tree traversal for upward pass
void upwardPass(struct Node * Ci) {
  for (struct Node * Cj=Ci->child; Cj!=Ci->child+Ci->numChilds; Cj++) {
    upwardPass(Cj);
  }
  if(Ci->numChilds==0) P2M(Ci);
  M2M(Ci);
}

//! Recursive call to dual tree traversal for horizontal pass
void horizontalPass(struct Node * Ci, struct Node * Cj, double theta) {
  double dX[3];
  for (int d=0; d<3; d++) dX[d] = Ci->X[d] - Cj->X[d];
  double R2 = (dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2]) * theta * theta;
  if (R2 > (Ci->R + Cj->R) * (Ci->R + Cj->R)) {
    M2P(Ci, Cj);
  } else if (Ci->numChilds == 0 && Cj->numChilds == 0) {
    P2P(Ci, Cj);
  } else {
    for (struct Node * cj=Cj->child; cj!=Cj->child+Cj->numChilds; cj++) {
      horizontalPass(Ci, cj, theta);
    }
  }
}

//! Recursive call to pre-order tree traversal for downward pass
void downwardPass(struct Node *Ci, struct Node * Cj, double theta) {
  if (Ci->numChilds==0) {
    horizontalPass(Ci, Cj, theta);
  }
  for (struct Node *C=Ci->child; C!=Ci->child+Ci->numChilds; C++) {
#pragma omp task untied if(C->numBodies > 100)
    downwardPass(C, Cj, theta);
  }
#pragma omp taskwait
}

//! Direct summation
void direct(struct Body * ibodies, int numTargets, struct Body * jbodies, int numBodies) {
  struct Node nodes[2];
  struct Node * Ci = &nodes[0];
  struct Node * Cj = &nodes[1];
  Ci->body = ibodies;
  Ci->numBodies = numTargets;
  Cj->body = jbodies;
  Cj->numBodies = numBodies;
  P2P(Ci, Cj);
}

int main(int argc, char ** argv) {
  int N = 10000;
  double theta = .4;
  double ncrit = 100;
  double G = 6.6743e-11;

  struct timeval tic, toc;
  gettimeofday(&tic, NULL);
  struct Body * bodies = (struct Body*)malloc(N*sizeof(struct Body));
#if 1
  initBodies(bodies, N);
#else
  FILE *file;
  if ( (file = fopen("initial.dat","rb")) == NULL ) {
    fprintf(stderr, "File open error.\n");
    exit(EXIT_FAILURE);
  }
  assert( fread(&N,sizeof(int),1,file) == 1 );
  double * x = (double*) malloc(N*sizeof(double));
  double * y = (double*) malloc(N*sizeof(double));
  double * z = (double*) malloc(N*sizeof(double));
  double * m = (double*) malloc(N*sizeof(double));
  double * ax = (double*) malloc(N*sizeof(double));
  double * ay = (double*) malloc(N*sizeof(double));
  double * az = (double*) malloc(N*sizeof(double));
  assert( fread(x,sizeof(double),N,file) == N );
  assert( fread(y,sizeof(double),N,file) == N );
  assert( fread(z,sizeof(double),N,file) == N );
  assert( fread(m,sizeof(double),N,file) == N );
  fclose(file);
  for (int b=0; b<N; b++) {
    bodies[b].X[0] = x[b];
    bodies[b].X[1] = y[b];
    bodies[b].X[2] = z[b];
    bodies[b].m = m[b];
    bodies[b].i = b;
  }
#endif
  double R0;
  double X0[3];
  getBounds(bodies, N, X0, &R0);
  struct Body * bodies2 = (struct Body*)malloc(N*sizeof(struct Body));
  struct Node * nodes = (struct Node*)malloc(N*(32/ncrit+1)*sizeof(struct Node));
  int numNodes = 1;
  buildTree(bodies, bodies2, 0, N, nodes, nodes, &numNodes, X0, R0, ncrit, false);
  upwardPass(nodes);
#pragma omp parallel
#pragma omp single nowait
  downwardPass(&nodes[0],&nodes[0],theta);
  gettimeofday(&toc, NULL);
  printf("FMM    : %g\n",timeDiff(tic,toc));

#if 0
  for (int b=0; b<N; b++) {
    int i = bodies[b].i;
    ax[i] = bodies[b].F[0] * G * bodies[b].m;
    ay[i] = bodies[b].F[1] * G * bodies[b].m;
    az[i] = bodies[b].F[2] * G * bodies[b].m;
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
#else
  gettimeofday(&tic, NULL);
  int numTargets = 100;
  int stride = N / numTargets;
  for (int b=0; b<numTargets; b++) {
    bodies2[b].m = bodies[b*stride].m;
    for (int d=0; d<3; d++) {
      bodies2[b].X[d] = bodies[b*stride].X[d];
      bodies2[b].F[d] = 0;
    }
  }
  direct(&bodies2[0], numTargets, &bodies[0], N);
  for (int b=0; b<numTargets; b++)
    for (int d=0; d<3; d++)
      bodies[b].F[d] = bodies[b*stride].F[d];
  gettimeofday(&toc, NULL);
  printf("Direct : %g\n",timeDiff(tic,toc));

  double diff = 0, norm = 0;
  for (int b=0; b<numTargets; b++) {
    diff += (bodies2[b].F[0] - bodies[b].F[0]) * (bodies2[b].F[0] - bodies[b].F[0]);
    diff += (bodies2[b].F[1] - bodies[b].F[1]) * (bodies2[b].F[1] - bodies[b].F[1]);
    diff += (bodies2[b].F[2] - bodies[b].F[2]) * (bodies2[b].F[2] - bodies[b].F[2]);
    norm += bodies[b].F[0] * bodies[b].F[0];
    norm += bodies[b].F[1] * bodies[b].F[1];
    norm += bodies[b].F[2] * bodies[b].F[2];
  }
  printf("Error  : %e\n", sqrtf(diff/norm));
#endif
  free(nodes);
  free(bodies);
  free(bodies2);
  return 0;
}
