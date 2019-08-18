#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

struct Body {
  double X[3];
  double q;
  double p;
  int index;
};
typedef std::vector<Body> Bodies;

struct Cell {
  int NCHILD;
  int NBODY;
  Cell * CHILD;
  Body * BODY;
  double X[3];
  double R;
  double M;
  double L;
  int index;
};
typedef std::vector<Cell> Cells;

void buildTree(Bodies & bodies, Cells & cells, Cell * cell, double * Xmin, double * X0, double R0, int begin, int end, int ncrit) {
  cell->BODY = &bodies[0] + begin;
  cell->NBODY = end - begin;
  cell->NCHILD = 0;
  cell->M = cell->L = 0;
  for (int d=0; d<3; d++) cell->X[d] = X0[d];
  cell->R = R0;
  if (end - begin < ncrit) return;
  // Count bodies in each octant
  int size[8] = {0};
  for (int b=begin; b<end; b++) {
    int octant = (bodies[b].X[0] > X0[0]) + ((bodies[b].X[1] > X0[1]) << 1) + ((bodies[b].X[2] > X0[2]) << 2);
    size[octant]++;
  }
  cell->NCHILD = 0;
  for (int i=0; i<8; i++)
    if (size[i]) cell->NCHILD++;
  // Calculate offset of each octant
  int counter[8];
  counter[0] = begin;
  for (int i=1; i<8; i++) {
    counter[i] = size[i-1] + counter[i-1];
  }
  // Sort bodies
  Bodies buffer = bodies;
  for (int b=begin; b<end; b++) {
    int octant = (buffer[b].X[0] > X0[0]) + ((buffer[b].X[1] > X0[1]) << 1) + ((buffer[b].X[2] > X0[2]) << 2);
    bodies[counter[octant]] = buffer[b];
    counter[octant]++;
  }
  cells.resize(cells.size()+cell->NCHILD);
  Cell * child = &cells.back() - cell->NCHILD + 1;
  cell->CHILD = child;
  // Calculate new center and radius
  double X[3], R;
  int c = 0;
  for (int i=0; i<8; i++) {
    R = R0 / 2;
    for (int d=0; d<3; d++) {
      X[d] = X0[d] + R * (((i & 1 << d) >> d) * 2 - 1);
    }
    // Recursive call only if size[i] != 0
    if (size[i]) {
      buildTree(bodies, cells, &child[c], Xmin, X, R, counter[i]-size[i], counter[i], ncrit);
      c++;
    }
  }
}

void P2M(Cell * cell) {
  for(int i=0; i<cell->NBODY; i++) {
    Body * body = cell->BODY+i;
    cell->M += body->q;
  }
}

void M2M(Cell * cell) {
  for(int i=0; i<cell->NCHILD; i++) {
    Cell * child = cell->CHILD+i;
    cell->M += child->M;
  }
}

void upwardPass(Cell * cell) {
  for(int i=0; i<cell->NCHILD; i++) {
    upwardPass(cell->CHILD+i);
  }
  if(cell->NCHILD==0) P2M(cell);
  else M2M(cell);
}

void M2L(Cell * icell, Cell * jcell) {
  icell->L += jcell->M;
}

void P2P(Cell * icell, Cell * jcell) {
  for (int i=0; i<icell->NBODY; i++) {
    Body * ibody = icell->BODY+i;
    for (int j=0; j<jcell->NBODY; j++) {
      Body * jbody = jcell->BODY+j;
      ibody->p += jbody->q;
    }
  }
}

void horizontalPass(Cell * icell, Cell * jcell) {
  double R = sqrtf((icell->X[0] - jcell->X[0]) * (icell->X[0] - jcell->X[0])
                   + (icell->X[1] - jcell->X[1]) * (icell->X[1] - jcell->X[1])
                   + (icell->X[2] - jcell->X[2]) * (icell->X[2] - jcell->X[2]));
  if (R > icell->R + jcell->R) {
    M2L(icell, jcell);
  }
  else if (icell->NCHILD == 0 && jcell->NCHILD == 0) {
    std::cout << icell->index << std::endl;
    P2P(icell, jcell);
  }
  else if (icell->R > jcell->R) {
    for (int i=0; i<icell->NCHILD; i++) horizontalPass(icell->CHILD+i,jcell);
  }
  else {
    for (int i=0; i<jcell->NCHILD; i++) horizontalPass(icell,jcell->CHILD+i);
  }
}

void L2L(Cell * cell) {
  for (int i=0; i<cell->NCHILD; i++) {
    Cell * child = cell->CHILD+i;
    child->L += cell->L;
  }
}

void L2P(Cell * cell) {
  for (int i=0; i<cell->NBODY; i++) {
    Body * body = cell->BODY+i;
    body->p += cell->L;
  }
}

void downwardPass(Cell * cell) {
  L2L(cell);
  if(cell->NCHILD==0) L2P(cell);
  for(int i=0; i<cell->NCHILD; i++) {
    downwardPass(cell->CHILD+i);
  }
}

int main(int argc, char ** argv) {
  int ncrit = 4;                                                // Number of bodies per leaf cell
  const int numBodies = 60;                                     // Number of bodies
  // Initialize bodies
  Bodies bodies(numBodies);
  for (int b=0; b<numBodies; b++) {                          // Loop over bodies
    for (int d=0; d<3; d++) {                                   //  Loop over dimension
      bodies[b].X[d] = drand48();                               //   Initialize coordinates
    }                                                           //  End loop over dimension
    bodies[b].q = 1;
    bodies[b].p = 0;
    bodies[b].index = b;
  }                                                             // End loop over bodies

  // Get bounds
  double Xmin[3], Xmax[3];
  for (int d=0; d<3; d++) {
    Xmin[d] = Xmax[d] = bodies[0].X[d];
  }
  for (int b=0; b<numBodies; b++) {
    for (int d=0; d<3; d++) {
      Xmin[d] = Xmin[d] > bodies[b].X[d] ? bodies[b].X[d] : Xmin[d];
      Xmax[d] = Xmax[d] < bodies[b].X[d] ? bodies[b].X[d] : Xmax[d];
    }
  }
  // Get center and radius
  double X0[3], R0;
  for (int d=0; d<3; d++) {
    X0[d] = (Xmin[d] + Xmax[d]) / 2;
  }
  R0 = std::max(Xmax[0] - Xmin[0], Xmax[1] - Xmin[1]);
  R0 = std::max(R0, Xmax[2] - Xmin[2]);
  R0 *= .50001;
  for (int d=0; d<3; d++) {
    Xmin[d] = X0[d] - R0;
    Xmax[d] = X0[d] + R0;
  }

  Cells cells(1);
  cells.reserve(numBodies);
  for (int d=0; d<3; d++) {
    cells[0].X[d] = X0[d];
  }
  cells[0].R = R0;
  buildTree(bodies, cells, &cells[0], Xmin, X0, R0, 0, numBodies, ncrit);
  upwardPass(&cells[0]);
  for (int i=0; i<cells.size(); i++) cells[i].index = i;
  horizontalPass(&cells[0],&cells[0]);
  downwardPass(&cells[0]);
  for (int i=0; i<bodies.size(); i++) {
    std::cout << i << " " << bodies[i].p << " " << std::endl;
  }
  return 0;
}
