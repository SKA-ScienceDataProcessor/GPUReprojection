#include <iostream>
#include <cmath>
#include <math.h>
#include <stdlib.h>

#define PRJERR_NULL_POINTER 1
#define PRJERR_BAD_PIX_SET_SIN 2
#define PRJERR_BAD_WORLD_SET_SIN 3

#define SIN 1

#define SIZEX 4096
#define SIZEY 4096
#define SIZE (SIZEX*SIZEY)

#define XLL 0.45
#define XUR 1.44
#define YLL -0.98
#define YUR 0.01

#define R2D (180/3.1415926)
#define D2R (3.1415926/180)

#define PAD_SIZE 2
#define IMG_SIZE 4096
#define IMG_PAD (IMG_SIZE+2*PAD_SIZE)
#define IMGX0 0.0
#define IMGX1 0.4
#define IMGY0 -0.4
#define IMGY1 0.0

#ifndef DATATYPE_INTERP
#define DATATYPE_INTERP DATATYPE
#define DATATYPE_INTERP2 DATATYPE2
#endif

__host__ __device__ float2 operator+(float2 in1, float2 in2) {
   float2 ret;
   ret.x = in1.x+in2.x;
   ret.y = in1.y+in2.y;
   return ret;
}
__host__ __device__ double2 operator+(double2 in1, double2 in2) {
   double2 ret;
   ret.x = in1.x+in2.x;
   ret.y = in1.y+in2.y;
   return ret;
}
__host__ __device__ double2 operator*(double2 in1, double in2) {
   double2 ret;
   ret.x = in1.x*in2;
   ret.y = in1.y*in2;
   return ret;
}
__host__ __device__ float2 operator*(float2 in1, float in2) {
   float2 ret;
   ret.x = in1.x*in2;
   ret.y = in1.y*in2;
   return ret;
}
__host__ __device__ double2 operator*(double in1, double2 in2) {
   double2 ret;
   ret.x = in1*in2.x;
   ret.y = in1*in2.y;
   return ret;
}
__host__ __device__ float2 operator*(float in1, float2 in2) {
   float2 ret;
   ret.x = in1*in2.x;
   ret.y = in1*in2.y;
   return ret;
}
void checkCudaError(int line, const char* filename) {
   cudaError_t err = cudaGetLastError();
   if (err) std::cerr << "Cuda error " << err << "(" << cudaGetErrorString(err) <<
                         ") on line " << line << " of file " << filename << std::endl;
}
struct prjprm {
   int flag;
   DATATYPE pv[3];
   DATATYPE x0, y0;
   DATATYPE w[4]; 
   DATATYPE r0;
   int bounds;
};

int sinx2s_alt(struct prjprm *prj, int nx, int ny, int sxy, int spt, DATATYPE *x, DATATYPE *y, DATATYPE *phi, DATATYPE *theta, int *stat)
//Alternate implementation of sinx2s as an intermediate step toward parallel GPU processing
{
  int status;
  const DATATYPE tol = 1.0e-13;
  DATATYPE a, b, c, d, eta, r2, sinth1, sinth2, sinthe, x0, xi, x1, xy, y0, y02,
         y1, z;
  int ix, iy;


  /* Initialize. */
  if (prj == 0x0) return PRJERR_NULL_POINTER;
  if (prj->flag != SIN) {
    return 4;
  }

  xi  = prj->pv[1];
  eta = prj->pv[2];

  status = 0;


  /* Do y dependence. */
  for (iy = 0; iy < ny; iy++) {
    y0 = (y[sxy*iy] + prj->y0)*prj->w[0];
    y02 = y0*y0;

    for (ix = 0; ix < nx; ix++) {
      
      x0 = (x[sxy*ix] + prj->x0)*prj->w[0];
      r2 = x0*x0 + y02;

      if (prj->w[1] == 0.0) {
        /* Orthographic projection. */
        if (r2 != 0.0) {
          phi[ix*spt+nx*iy*spt] = atan2(x0, -y0);
        } else {
          phi[ix*spt+nx*iy*spt] = 0.0;
        }

        if (r2 < 0.5) {
          theta[ix*spt+nx*iy*spt] = acos(sqrt(r2));
        } else if (r2 <= 1.0) {
          theta[ix*spt+nx*iy*spt] = asin(sqrt(1.0 - r2));
        } else {
          stat[ix*spt+nx*iy*spt] = 1;
          if (!status) status = PRJERR_BAD_PIX_SET_SIN;
          continue;
        }

      } else {
        /* "Synthesis" projection. */
        xy = x0*xi + y0*eta;

        if (r2 < 1.0e-10) {
          /* Use small angle formula. */
          z = r2/2.0;
          theta[ix*spt+nx*iy*spt] = 90.0 - R2D*sqrt(r2/(1.0 + xy));

        } else {
          a = prj->w[2];
          b = xy - prj->w[1];
          c = r2 - xy - xy + prj->w[3];
          d = b*b - a*c;

          /* Check for a solution. */
          if (d < 0.0) {
            phi[ix*spt+nx*iy*spt] = 0.0;
            theta[ix*spt+nx*iy*spt] = 0.0;
            stat[ix*spt+nx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            continue;
          }
          d = sqrt(d);

          /* Choose solution closest to pole. */
          sinth1 = (-b + d)/a;
          sinth2 = (-b - d)/a;
          sinthe = (sinth1 > sinth2) ? sinth1 : sinth2;
          if (sinthe > 1.0) {
            if (sinthe-1.0 < tol) {
              sinthe = 1.0;
            } else {
              sinthe = (sinth1 < sinth2) ? sinth1 : sinth2;
            }
          }

          if (sinthe < -1.0) {
            if (sinthe+1.0 > -tol) {
              sinthe = -1.0;
            }
          }

          if (sinthe > 1.0 || sinthe < -1.0) {
            phi[ix*spt+nx*iy*spt] = 0.0;
            theta[ix*spt+nx*iy*spt] = 0.0;
            stat[ix*spt+nx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            continue;
          }

          theta[ix*spt+nx*iy*spt] = asin(sinthe);
          z = 1.0 - sinthe;
        }

        x1 = -y0 + eta*z;
        y1 =  x0 -  xi*z;
        if (x1 == 0.0 && y1 == 0.0) {
          phi[ix*spt+nx*iy*spt] = 0.0;
        } else {
          phi[ix*spt+nx*iy*spt] = atan2(y1,x1);
        }
      }

      stat[ix*spt+nx*iy*spt] = 0;
    }
  }

  return 0;
}
//__device__
int sinx2s(struct prjprm *prj, int nx, int ny, int sxy, int spt, DATATYPE *x, DATATYPE *y, DATATYPE *phi, DATATYPE *theta, int *stat)
{
  int mx, my, rowlen, rowoff, status;
  const DATATYPE tol = 1.0e-13;
  DATATYPE a, b, c, d, eta, r2, sinth1, sinth2, sinthe, x0, xi, x1, xy, y0, y02,
         y1, z;
  int ix, iy, *statp;
  const DATATYPE *xp, *yp;
  DATATYPE *phip, *thetap;


  /* Initialize. */
  if (prj == 0x0) return PRJERR_NULL_POINTER;
  if (prj->flag != SIN) {
    return 4;
  }

  xi  = prj->pv[1];
  eta = prj->pv[2];

  if (ny > 0) {
    mx = nx;
    my = ny;
  } else {
    mx = 1;
    my = 1;
    ny = nx;
  }

  status = 0;


  /* Do x dependence. */
  xp = x;
  rowoff = 0;
  rowlen = nx*spt;
  for (ix = 0; ix < nx; ix++, rowoff += spt, xp += sxy) {
    x0 = (*xp + prj->x0)*prj->w[0];

    phip = phi + rowoff;
    for (iy = 0; iy < my; iy++) {
      *phip = x0;
      phip += rowlen;
    }
  }


  /* Do y dependence. */
  yp = y;
  phip   = phi;
  thetap = theta;
  statp  = stat;
  for (iy = 0; iy < ny; iy++, yp += sxy) {
    y0 = (*yp + prj->y0)*prj->w[0];
    y02 = y0*y0;

    for (ix = 0; ix < mx; ix++, phip += spt, thetap += spt) {
      /* Compute intermediaries. */
      x0 = *phip;
      r2 = x0*x0 + y02;

      if (prj->w[1] == 0.0) {
        /* Orthographic projection. */
        if (r2 != 0.0) {
          *phip = atan2(x0, -y0);
        } else {
          *phip = 0.0;
        }

        if (r2 < 0.5) {
          *thetap = acos(sqrt(r2));
        } else if (r2 <= 1.0) {
          *thetap = asin(sqrt(1.0 - r2));
        } else {
          *(statp++) = 1;
          if (!status) status = PRJERR_BAD_PIX_SET_SIN;
          continue;
        }

      } else {
        /* "Synthesis" projection. */
        xy = x0*xi + y0*eta;

        if (r2 < 1.0e-10) {
          /* Use small angle formula. */
          z = r2/2.0;
          *thetap = 90.0 - R2D*sqrt(r2/(1.0 + xy));

        } else {
          a = prj->w[2];
          b = xy - prj->w[1];
          c = r2 - xy - xy + prj->w[3];
          d = b*b - a*c;

          /* Check for a solution. */
          if (d < 0.0) {
            *phip = 0.0;
            *thetap = 0.0;
            *(statp++) = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            continue;
          }
          d = sqrt(d);

          /* Choose solution closest to pole. */
          sinth1 = (-b + d)/a;
          sinth2 = (-b - d)/a;
          sinthe = (sinth1 > sinth2) ? sinth1 : sinth2;
          if (sinthe > 1.0) {
            if (sinthe-1.0 < tol) {
              sinthe = 1.0;
            } else {
              sinthe = (sinth1 < sinth2) ? sinth1 : sinth2;
            }
          }

          if (sinthe < -1.0) {
            if (sinthe+1.0 > -tol) {
              sinthe = -1.0;
            }
          }

          if (sinthe > 1.0 || sinthe < -1.0) {
            *phip = 0.0;
            *thetap = 0.0;
            *(statp++) = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            continue;
          }

          *thetap = asin(sinthe);
          z = 1.0 - sinthe;
        }

        x1 = -y0 + eta*z;
        y1 =  x0 -  xi*z;
        if (x1 == 0.0 && y1 == 0.0) {
          *phip = 0.0;
        } else {
          *phip = atan2(y1,x1);
        }
      }

      *(statp++) = 0;
    }
  }

  return 0;
}

/*--------------------------------------------------------------------------*/
int sins2x_alt(prjprm *prj, int nphi, int ntheta, int spt, int sxy, DATATYPE *phi, DATATYPE *theta, DATATYPE *x, DATATYPE *y, int *stat)

{
  int mphi, status;
  DATATYPE cosphi, costhe, sinphi, r, t, z, z1, z2;
  register int iphi, itheta, istat;


  status = 0;

  /* Do theta dependence. */
  for (itheta = 0; itheta < ntheta; itheta++) {
    t = (90.0 - fabs(theta[itheta*spt]))*D2R;
    if (t < 1.0e-5) {
      if (theta[itheta*spt] > 0.0) {
         z = t*t/2.0;
      } else {
         z = 2.0 - t*t/2.0;
      }
      costhe = t;
    } else {
      z = 1.0 - sin(theta[itheta*spt]);
      costhe = cos(theta[itheta*spt]);
    }
    r = prj->r0*costhe;

    if (prj->w[1] == 0.0) {
      /* Orthographic projection. */
      istat = 0;
      if (prj->bounds&1) {
        if (theta[itheta*spt] < 0.0) {
          istat = 1;
          if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
        }
      }

      for (iphi = 0; iphi < mphi; iphi++) {
        sincos(phi[iphi*sxy], &sinphi, &cosphi); 
        x[iphi*sxy+mphi*sxy*itheta] =  r*sinphi - prj->x0;
        y[iphi*sxy+mphi*sxy*itheta] =  -r*cosphi - prj->y0;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      }

    } else {
      /* "Synthesis" projection. */
      z *= prj->r0;
      z1 = prj->pv[1]*z - prj->x0;
      z2 = prj->pv[2]*z - prj->y0;

      for (iphi = 0; iphi < mphi; iphi++) {
        istat = 0;
        sincos(phi[iphi*sxy], &sinphi, &cosphi); 
        if (prj->bounds&1) {
          t = -atan(prj->pv[1]*(sinphi) - prj->pv[2]*(cosphi));
          if (theta[itheta*spt] < t) {
            istat = 1;
            if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
          }
        }

        x[iphi*sxy+mphi*sxy*itheta] = r*sinphi + z1;
        y[iphi*sxy+mphi*sxy*itheta] = -r*cosphi + z2;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      }
    }
  }

  return status;
}
int sins2x_alt2(prjprm *prj, int nphi, int ntheta, int spt, int sxy, DATATYPE *phi, DATATYPE *theta, DATATYPE *x, DATATYPE *y, int *stat)

{
  int status;
  DATATYPE cosphi, costhe, sinphi, r, t, z, z1, z2;
  register int iphi, itheta, istat;


  status = 0;

  /* Do theta dependence. */
  for (itheta = 0; itheta < ntheta; itheta++) {
  for (iphi = 0; iphi < nphi; iphi++) {
    t = (90.0 - fabs(theta[iphi*spt+nphi*itheta*spt]))*D2R;
    if (t < 1.0e-5) {
      if (theta[iphi*spt+nphi*itheta*spt] > 0.0) {
         z = t*t/2.0;
      } else {
         z = 2.0 - t*t/2.0;
      }
      costhe = t;
    } else {
      z = 1.0 - sin(theta[iphi*spt+nphi*itheta*spt]);
      costhe = cos(theta[iphi*spt+nphi*itheta*spt]);
    }
    r = prj->r0*costhe;

    if (prj->w[1] == 0.0) {
      /* Orthographic projection. */
      istat = 0;
      if (prj->bounds&1) {
        if (theta[iphi*sxy+nphi*itheta*sxy] < 0.0) {
          istat = 1;
          if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
        }
      }

        sincos(phi[iphi*sxy+nphi*itheta*sxy], &sinphi, &cosphi); 
        x[iphi*sxy+nphi*sxy*itheta] =  r*sinphi - prj->x0;
        y[iphi*sxy+nphi*sxy*itheta] =  -r*cosphi - prj->y0;
        stat[iphi*sxy+nphi*sxy*itheta] = istat;
    } else {
      /* "Synthesis" projection. */
      z *= prj->r0;
      z1 = prj->pv[1]*z - prj->x0;
      z2 = prj->pv[2]*z - prj->y0;

        istat = 0;
        sincos(phi[iphi*sxy+nphi*itheta*sxy], &sinphi, &cosphi); 
        if (prj->bounds&1) {
          t = -atan(prj->pv[1]*(sinphi) - prj->pv[2]*(cosphi));
          if (theta[iphi*spt+nphi*itheta*spt] < t) {
            istat = 1;
            if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
          }
        }

        x[iphi*sxy+nphi*sxy*itheta] = r*sinphi + z1;
        y[iphi*sxy+nphi*sxy*itheta] = -r*cosphi + z2;
        stat[iphi*sxy+nphi*sxy*itheta] = istat;
      }
    }
  }

  return status;
}
int sins2x(prjprm *prj, int nphi, int ntheta, int spt, int sxy, DATATYPE *phi, DATATYPE *theta, DATATYPE *x, DATATYPE *y, int *stat)

{
  int mphi, mtheta, rowlen, rowoff, status;
  DATATYPE cosphi, costhe, sinphi, r, t, z, z1, z2;
  register int iphi, itheta, istat, *statp;
  register const DATATYPE *phip, *thetap;
  register DATATYPE *xp, *yp;


  if (ntheta > 0) {
    mphi   = nphi;
    mtheta = ntheta;
  } else {
    mphi   = 1;
    mtheta = 1;
    ntheta = nphi;
  }

  status = 0;


  /* Do phi dependence. */
  phip = phi;
  rowoff = 0;
  rowlen = nphi*sxy;
  for (iphi = 0; iphi < nphi; iphi++, rowoff += sxy, phip += spt) {
    sincos(*phip, &sinphi, &cosphi);

    xp = x + rowoff;
    yp = y + rowoff;
    for (itheta = 0; itheta < mtheta; itheta++) {
      *xp = sinphi;
      *yp = cosphi;
      xp += rowlen;
      yp += rowlen;
    }
  }


  /* Do theta dependence. */
  thetap = theta;
  xp = x;
  yp = y;
  statp = stat;
  for (itheta = 0; itheta < ntheta; itheta++, thetap += spt) {
    t = (90.0 - fabs(*thetap))*D2R;
    if (t < 1.0e-5) {
      if (*thetap > 0.0) {
         z = t*t/2.0;
      } else {
         z = 2.0 - t*t/2.0;
      }
      costhe = t;
    } else {
      z = 1.0 - sin(*thetap);
      costhe = cos(*thetap);
    }
    r = prj->r0*costhe;

    if (prj->w[1] == 0.0) {
      /* Orthographic projection. */
      istat = 0;
      if (prj->bounds&1) {
        if (*thetap < 0.0) {
          istat = 1;
          if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
        }
      }

      for (iphi = 0; iphi < mphi; iphi++, xp += sxy, yp += sxy) {
        *xp =  r*(*xp) - prj->x0;
        *yp = -r*(*yp) - prj->y0;
        *(statp++) = istat;
      }

    } else {
      /* "Synthesis" projection. */
      z *= prj->r0;
      z1 = prj->pv[1]*z - prj->x0;
      z2 = prj->pv[2]*z - prj->y0;

      for (iphi = 0; iphi < mphi; iphi++, xp += sxy, yp += sxy) {
        istat = 0;
        if (prj->bounds&1) {
          t = -atan(prj->pv[1]*(*xp) - prj->pv[2]*(*yp));
          if (*thetap < t) {
            istat = 1;
            if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
          }
        }

        *xp =  r*(*xp) + z1;
        *yp = -r*(*yp) + z2;
        *(statp++) = istat;
      }
    }
  }

  return status;
}
__device__
void sinx2s_dev(DATATYPE xi, DATATYPE eta, DATATYPE xoff, DATATYPE yoff, DATATYPE scale, 
                DATATYPE aoff, DATATYPE boff, DATATYPE coff, int nx, int ny, 
                int sxy, int spt, DATATYPE *x, DATATYPE *y, DATATYPE *phi, 
                DATATYPE *theta, int *stat)
{
  int mx, my, status;
  const DATATYPE tol = 1.0e-13;
  DATATYPE a, b, c, d, r2, sinth1, sinth2, sinthe, x0, x1, xy, y0, y02,
         y1, z;
  int ix, iy;


  mx = nx;
  my = ny;

  status = 0;


  /* Do x dependence. */

  /* Do y dependence. */
  iy=threadIdx.y + blockIdx.y * blockDim.y;
  ix=threadIdx.x + blockIdx.x * blockDim.x;
  if (ix>mx || iy>my) return;
    y0 = (y[sxy*iy] + yoff)*scale;
    y02 = y0*y0;

      /* Compute intermediaries. */
      x0 = (x[sxy*ix] + xoff)*scale;
      r2 = x0*x0 + y02;

      if (boff == 0.0) {
        /* Orthographic projection. */
        if (r2 != 0.0) {
          phi[ix*spt+mx*iy*spt] = atan2(x0, -y0);
        } else {
          phi[ix*spt+mx*iy*spt] = 0.0;
        }

        if (r2 < 0.5) {
          theta[ix*spt+mx*iy*spt] = acos(sqrt(r2));
        } else if (r2 <= 1.0) {
          theta[ix*spt+mx*iy*spt] = asin(sqrt(1.0 - r2));
        } else {
          stat[ix*spt+mx*iy*spt] = 1;
          if (!status) status = PRJERR_BAD_PIX_SET_SIN;
          return;
        }

      } else {
        /* "Synthesis" projection. */
        xy = x0*xi + y0*eta;

        if (r2 < 1.0e-10) {
          /* Use small angle formula. */
          z = r2/2.0;
          theta[ix*spt+mx*iy*spt] = 90.0 - R2D*sqrt(r2/(1.0 + xy));

        } else {
          a = aoff;
          b = xy + boff;
          c = r2 - xy - xy + coff;
          d = b*b - a*c;

          /* Check for a solution. */
          if (d < 0.0) {
            phi[ix*spt+mx*iy*spt] = 0.0;
            theta[ix*spt+mx*iy*spt] = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }
          d = sqrt(d);

          /* Choose solution closest to pole. */
          sinth1 = (-b + d)/a;
          sinth2 = (-b - d)/a;
          sinthe = (sinth1 > sinth2) ? sinth1 : sinth2;
          if (sinthe > 1.0) {
            if (sinthe-1.0 < tol) {
              sinthe = 1.0;
            } else {
              sinthe = (sinth1 < sinth2) ? sinth1 : sinth2;
            }
          }

          if (sinthe < -1.0) {
            if (sinthe+1.0 > -tol) {
              sinthe = -1.0;
            }
          }

          if (sinthe > 1.0 || sinthe < -1.0) {
            phi[ix*spt+mx*iy*spt] = 0.0;
            theta[ix*spt+mx*iy*spt] = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }

          theta[ix*spt+mx*iy*spt] = 0.0;
           asin(sinthe);
          z = 1.0 - sinthe;
        }

        x1 = -y0 + eta*z;
        y1 =  x0 -  xi*z;
        if (x1 == 0.0 && y1 == 0.0) {
          phi[ix*spt+mx*iy*spt] = 0.0;
        } else {
          phi[ix*spt+mx*iy*spt] = atan2(y1,x1);
        }
      }
   return;
}
__device__
void sins2x_dev(DATATYPE r0, DATATYPE scale, DATATYPE x0, DATATYPE y0, DATATYPE sintheta0,
               DATATYPE costheta0, int bounds, int nphi, int ntheta, int spt, 
               int sxy, DATATYPE *phi, DATATYPE *theta, DATATYPE *x, DATATYPE *y, int *stat)

{
  int mphi, mtheta, status;
  DATATYPE cosphi, costhe, sinphi, r, t, z, z1, z2;
  register int iphi, itheta, istat;


  mphi   = nphi;
  mtheta = ntheta;

  status = 0;

  /* Do theta dependence. */
  itheta = threadIdx.y + blockIdx.y * blockDim.y;
  iphi = threadIdx.x + blockIdx.x * blockDim.x;
  if (iphi>mphi || itheta>mtheta) return;
  //for (itheta = 0; itheta < ntheta; itheta++) {
    t = (90.0 - fabs(theta[itheta*spt]))*D2R;
    if (t < 1.0e-5) {
      if (theta[itheta*spt] > 0.0) {
         z = t*t/2.0;
      } else {
         z = 2.0 - t*t/2.0;
      }
      costhe = t;
    } else {
      z = 1.0 - sin(theta[itheta*spt]);
      costhe = cos(theta[itheta*spt]);
    }
    r = r0*costhe;

    if (scale == 0.0) {
      /* Orthographic projection. */
      istat = 0;
      if (bounds&1) {
        if (theta[itheta*spt] < 0.0) {
          istat = 1;
          if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
        }
      }

      //for (iphi = 0; iphi < mphi; iphi++, xp += sxy, yp += sxy) {
        sincos(phi[iphi*sxy], &sinphi, &cosphi); 
        x[iphi*sxy+mphi*sxy*itheta] =  r*sinphi - x0;
        y[iphi*sxy+mphi*sxy*itheta] =  -r*cosphi - y0;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}

    } else {
      /* "Synthesis" projection. */
      z *= r0;
      z1 = sintheta0*z - x0;
      z2 = costheta0*z - y0;

      //for (iphi = 0; iphi < mphi; iphi++) {
        istat = 0;
        sincos(phi[iphi*sxy], &sinphi, &cosphi); 
        if (bounds&1) {
          t = -atan(sintheta0*(sinphi) - costheta0*(cosphi));
          if (theta[itheta*spt] < t) {
            istat = 1;
            if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
          }
        }

        x[iphi*sxy+mphi*sxy*itheta] = r*sinphi + z1;
        y[iphi*sxy+mphi*sxy*itheta] = -r*cosphi + z2;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}
    }
  //}

  //return status;
}
__device__
void cc_dev(DATATYPE xi, DATATYPE eta, DATATYPE xoff, DATATYPE yoff, DATATYPE scale, 
                DATATYPE aoff, DATATYPE boff, DATATYPE coff, int nx, int ny, int nphi, int ntheta, 
                int sxy, int spt, DATATYPE r0, DATATYPE scale_out, DATATYPE xoff_out, DATATYPE yoff_out,
                DATATYPE sintheta0, DATATYPE costheta0, int bounds, DATATYPE *x, DATATYPE *y, int *stat,
                DATATYPE* phi, DATATYPE* theta)
{
  int mx, my, status;
  const DATATYPE tol = 1.0e-13;
  DATATYPE a, b, c, d, r2, sinth1, sinth2, sinthe, x0, x1, xy, y0, y02,
         y1, z;
  int ix, iy;
  DATATYPE lphi, ltheta;


  mx = nx;
  my = ny;

  status = 0;


  /* Do x dependence. */

  /* Do y dependence. */
  iy=threadIdx.y + blockIdx.y * blockDim.y;
  ix=threadIdx.x + blockIdx.x * blockDim.x;
  if (ix>mx || iy>my) return;
    y0 = (y[sxy*iy] + yoff)*scale;
    y02 = y0*y0;

      /* Compute intermediaries. */
      x0 = (x[sxy*ix] + xoff)*scale;
      r2 = x0*x0 + y02;

      if (boff == 0.0) {
        /* Orthographic projection. */
        if (r2 != 0.0) {
          lphi = atan2(x0, -y0);
        } else {
          lphi = 0.0;
        }

        if (r2 < 0.5) {
          ltheta = acos(sqrt(r2));
        } else if (r2 <= 1.0) {
          ltheta = asin(sqrt(1.0 - r2));
        } else {
          stat[ix*spt+mx*iy*spt] = 1;
          if (!status) status = PRJERR_BAD_PIX_SET_SIN;
          return;
        }

      } else {
        /* "Synthesis" projection. */
        xy = x0*xi + y0*eta;

        if (r2 < 1.0e-10) {
          /* Use small angle formula. */
          z = r2/2.0;
          ltheta = 90.0 - R2D*sqrt(r2/(1.0 + xy));

        } else {
          a = aoff;
          b = xy + boff;
          c = r2 - xy - xy + coff;
          d = b*b - a*c;

          /* Check for a solution. */
          if (d < 0.0) {
            lphi = 0.0;
            ltheta = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }
          d = sqrt(d);

          /* Choose solution closest to pole. */
          sinth1 = (-b + d)/a;
          sinth2 = (-b - d)/a;
          sinthe = (sinth1 > sinth2) ? sinth1 : sinth2;
          if (sinthe > 1.0) {
            if (sinthe-1.0 < tol) {
              sinthe = 1.0;
            } else {
              sinthe = (sinth1 < sinth2) ? sinth1 : sinth2;
            }
          }

          if (sinthe < -1.0) {
            if (sinthe+1.0 > -tol) {
              sinthe = -1.0;
            }
          }

          if (sinthe > 1.0 || sinthe < -1.0) {
            lphi = 0.0;
            ltheta = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }

          ltheta = asin(sinthe);
          z = 1.0 - sinthe;
        }

        x1 = -y0 + eta*z;
        y1 =  x0 -  xi*z;
        if (x1 == 0.0 && y1 == 0.0) {
          lphi = 0.0;
        } else {
          lphi = atan2(y1,x1);
        }
      }
  int mphi, mtheta;
  DATATYPE cosphi, costhe, sinphi, r, t, z1, z2;
  register int iphi, itheta, istat;


  mphi   = nphi;
  mtheta = ntheta;

  status = 0;

  /* Do theta dependence. */
  itheta = threadIdx.y + blockIdx.y * blockDim.y;
  iphi = threadIdx.x + blockIdx.x * blockDim.x;
  if (iphi>mphi || itheta>mtheta) return;
  //for (itheta = 0; itheta < ntheta; itheta++) {
    t = (90.0 - fabs(ltheta))*D2R;
    if (t < 1.0e-5) {
      if (ltheta > 0.0) {
         z = t*t/2.0;
      } else {
         z = 2.0 - t*t/2.0;
      }
      costhe = t;
    } else {
      z = 1.0 - sin(ltheta);
      costhe = cos(ltheta);
    }
    r = r0*costhe;

    if (scale_out == 0.0) {
      /* Orthographic projection. */
      istat = 0;
      if (bounds&1) {
        if (ltheta < 0.0) {
          istat = 1;
          if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
        }
      }

      //for (iphi = 0; iphi < mphi; iphi++, xp += sxy, yp += sxy) {
        sincos(lphi, &sinphi, &cosphi); 
        x[iphi*sxy+mphi*sxy*itheta] =  r*sinphi - xoff_out;
        y[iphi*sxy+mphi*sxy*itheta] =  -r*cosphi - yoff_out;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}

    } else {
      /* "Synthesis" projection. */
      z *= r0;
      z1 = sintheta0*z - xoff_out;
      z2 = costheta0*z - yoff_out;

      //for (iphi = 0; iphi < mphi; iphi++) {
        istat = 0;
        sincos(lphi, &sinphi, &cosphi); 
        if (bounds&1) {
          t = -atan(sintheta0*(sinphi) - costheta0*(cosphi));
          if (ltheta < t) {
            istat = 1;
            if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
          }
        }

        x[iphi*sxy+mphi*sxy*itheta] = r*sinphi + z1;
        y[iphi*sxy+mphi*sxy*itheta] = -r*cosphi + z2;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}
    }
  //}

  //return status;
}
__device__
void reproject_dev(DATATYPE xi, DATATYPE eta, DATATYPE xoff, DATATYPE yoff, DATATYPE scale, 
                DATATYPE aoff, DATATYPE boff, DATATYPE coff, int nx, int ny, int nphi, int ntheta, 
                int sxy, int spt, DATATYPE r0, DATATYPE scale_out, DATATYPE xoff_out, DATATYPE yoff_out,
                DATATYPE sintheta0, DATATYPE costheta0, int bounds, DATATYPE* x_in, DATATYPE* y_in, 
#ifdef __USE_TEX
                cudaTextureObject_t img_orig, 
#else
                DATATYPE_INTERP2* img_orig,
#endif
                int sz, DATATYPE xgrid, DATATYPE ygrid, 
                DATATYPE_INTERP2* img_out, int *stat)
{
  int mx, my, status;
  const DATATYPE tol = 1.0e-13;
  DATATYPE a, b, c, d, r2, sinth1, sinth2, sinthe, x0, x1, xy, y0, y02,
         y1, z, x, y;
  int ix, iy;
  DATATYPE lphi, ltheta;


  mx = nx;
  my = ny;

  status = 0;


  /* Do x dependence. */

  /* Do y dependence. */
  iy=threadIdx.y + blockIdx.y * blockDim.y;
  ix=threadIdx.x + blockIdx.x * blockDim.x;
  if (ix>mx || iy>my) return;
    //Assumes a regular, rectangular grid
    y0 = (y_in[sxy*iy] + yoff)*scale;
    y02 = y0*y0;

      /* Compute intermediaries. */
      x0 = (x_in[sxy*ix] + xoff)*scale;
      r2 = x0*x0 + y02;

      if (boff == 0.0) {
        /* Orthographic projection. */
        if (r2 != 0.0) {
          lphi = atan2(x0, -y0);
        } else {
          lphi = 0.0;
        }

        if (r2 < 0.5) {
          ltheta = acos(sqrt(r2));
        } else if (r2 <= 1.0) {
          ltheta = asin(sqrt(1.0 - r2));
        } else {
          stat[ix*spt+mx*iy*spt] = 1;
          if (!status) status = PRJERR_BAD_PIX_SET_SIN;
          return;
        }

      } else {
        /* "Synthesis" projection. */
        xy = x0*xi + y0*eta;

        if (r2 < 1.0e-10) {
          /* Use small angle formula. */
          z = r2/2.0;
          ltheta = 90.0 - R2D*sqrt(r2/(1.0 + xy));

        } else {
          a = aoff;
          b = xy + boff;
          c = r2 - xy - xy + coff;
          d = b*b - a*c;

          /* Check for a solution. */
          if (d < 0.0) {
            lphi = 0.0;
            ltheta = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }
          d = sqrt(d);

          /* Choose solution closest to pole. */
          sinth1 = (-b + d)/a;
          sinth2 = (-b - d)/a;
          sinthe = (sinth1 > sinth2) ? sinth1 : sinth2;
          if (sinthe > 1.0) {
            if (sinthe-1.0 < tol) {
              sinthe = 1.0;
            } else {
              sinthe = (sinth1 < sinth2) ? sinth1 : sinth2;
            }
          }

          if (sinthe < -1.0) {
            if (sinthe+1.0 > -tol) {
              sinthe = -1.0;
            }
          }

          if (sinthe > 1.0 || sinthe < -1.0) {
            lphi = 0.0;
            ltheta = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }

          ltheta = asin(sinthe);
          z = 1.0 - sinthe;
        }

        x1 = -y0 + eta*z;
        y1 =  x0 -  xi*z;
        if (x1 == 0.0 && y1 == 0.0) {
          lphi = 0.0;
        } else {
          lphi = atan2(y1,x1);
        }
      }
  int mphi, mtheta;
  DATATYPE cosphi, costhe, sinphi, r, t, z1, z2;
  int iphi, itheta, istat;


  mphi   = nphi;
  mtheta = ntheta;

  status = 0;

  /* Do theta dependence. */
  itheta = threadIdx.y + blockIdx.y * blockDim.y;
  iphi = threadIdx.x + blockIdx.x * blockDim.x;
  if (iphi>mphi || itheta>mtheta) return;
  //for (itheta = 0; itheta < ntheta; itheta++) {
    t = (90.0 - fabs(ltheta))*D2R;
    if (t < 1.0e-5) {
      if (ltheta > 0.0) {
         z = t*t/2.0;
      } else {
         z = 2.0 - t*t/2.0;
      }
      costhe = t;
    } else {
      z = 1.0 - sin(ltheta);
      costhe = cos(ltheta);
    }
    r = r0*costhe;

    if (scale_out == 0.0) {
      /* Orthographic projection. */
      istat = 0;
      if (bounds&1) {
        if (ltheta < 0.0) {
          istat = 1;
          if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
        }
      }

      //for (iphi = 0; iphi < mphi; iphi++, xp += sxy, yp += sxy) {
        sincos(lphi, &sinphi, &cosphi); 
        x =  r*sinphi - xoff_out;
        y =  -r*cosphi - yoff_out;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}

    } else {
      /* "Synthesis" projection. */
      z *= r0;
      z1 = sintheta0*z - xoff_out;
      z2 = costheta0*z - yoff_out;

      //for (iphi = 0; iphi < mphi; iphi++) {
        istat = 0;
        sincos(lphi, &sinphi, &cosphi); 
        if (bounds&1) {
          t = -atan(sintheta0*(sinphi) - costheta0*(cosphi));
          if (ltheta < t) {
            istat = 1;
            if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
          }
        }

        x = r*sinphi + z1;
        y = -r*cosphi + z2;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}
    }
  //}

    DATATYPE thisx = x-IMGX0;
    DATATYPE thisy = y-IMGY0;
    x0 = floorf(thisx/xgrid)+PAD_SIZE;
    DATATYPE xfrac = thisx/xgrid-x0+PAD_SIZE;
    y0 = floorf(thisy/ygrid)+PAD_SIZE;
    DATATYPE yfrac = thisy/ygrid-y0+PAD_SIZE;
    int inx0 = IMG_PAD*y0+x0;
    inx0 %= IMG_PAD*IMG_PAD;

#ifdef __USE_TEX
#define FETCHX(A,B) tex1Dfetch<float>(A,2*(B))
#define FETCHY(A,B) tex1Dfetch<float>(A,2*(B)+1)
#else
#define FETCHX(A,B) A[B].x
#define FETCHY(A,B) A[B].y
#endif

#if __INTERP_LINEAR
    DATATYPE_INTERP out_x = FETCHX(img_orig,inx0);
    DATATYPE_INTERP out_y = FETCHY(img_orig,inx0);
    out_x *= (1-xfrac)*(1-yfrac);
    out_y *= (1-xfrac)*(1-yfrac);
    out_x += (1-xfrac)*yfrac*FETCHX(img_orig,inx0+IMG_PAD);
    out_y += (1-xfrac)*yfrac*FETCHY(img_orig,inx0+IMG_PAD);
    out_x += xfrac*(1-yfrac)*FETCHX(img_orig,inx0+1);
    out_y += xfrac*(1-yfrac)*FETCHY(img_orig,inx0+1);
    out_x += xfrac*yfrac*FETCHX(img_orig,inx0+IMG_PAD+1);
    out_y += xfrac*yfrac*FETCHY(img_orig,inx0+IMG_PAD+1);
    img_out[iphi+mphi*itheta].x = out_x;
    img_out[iphi+mphi*itheta].y = out_y;
#else
    DATATYPE w[4];
    w[0] = (-xfrac*xfrac*xfrac + 3*xfrac*xfrac - 3*xfrac + 1)/6.0;
    w[1] = (3*xfrac*xfrac*xfrac - 6*xfrac*xfrac + 4)/6.0;
    w[2] = (-3*xfrac*xfrac*xfrac + 3*xfrac*xfrac + 3*xfrac + 1)/6.0;
    w[3] = xfrac*xfrac*xfrac/6.0;
    DATATYPE_INTERP2 ivals[4];
    inx0 -= IMG_PAD;
    for (int qq=0;qq<4;qq++,inx0+=IMG_PAD) {
       ivals[qq] = img_orig[inx0-1]*w[0]
                 + img_orig[inx0]*w[1]
                 + img_orig[inx0+1]*w[2]
                 + img_orig[inx0+2]*w[3];
    }
    DATATYPE_INTERP2 out = ivals[0]*(-yfrac*yfrac*yfrac+3*yfrac*yfrac-3*yfrac+1) +
                 ivals[1]*(3*yfrac*yfrac*yfrac-6*yfrac*yfrac-4) +
                 ivals[2]*(-3*yfrac*yfrac*yfrac+3*yfrac*yfrac+3*yfrac+1) +
                 ivals[3]*yfrac*yfrac*yfrac;
    img_out[iphi+mphi*itheta] = out *(1.0/6.0);
   
#endif
  //return status;
}
__device__
void coord_convert_dev(DATATYPE xi, DATATYPE eta, DATATYPE xoff, DATATYPE yoff, DATATYPE scale, 
                DATATYPE aoff, DATATYPE boff, DATATYPE coff, int nx, int ny, 
                int sxy, int spt, DATATYPE r0, DATATYPE scale_out, DATATYPE xoff_out, DATATYPE yoff_out,
                DATATYPE sintheta0, DATATYPE costheta0, int bounds, DATATYPE *x, DATATYPE *y, int *stat)
{
  int mx, my, status;
  const DATATYPE tol = 1.0e-13;
  DATATYPE a, b, c, d, r2, sinth1, sinth2, sinthe, x0, x1, xy, y0, y02,
         y1, z;
  int ix, iy;
  DATATYPE phi, theta;


  mx = nx;
  my = ny;

  status = 0;


  /* Do x dependence. */

  /* Do y dependence. */
  iy=threadIdx.y + blockIdx.y * blockDim.y;
  ix=threadIdx.x + blockIdx.x * blockDim.x;
  if (ix>mx || iy>my) return;
    y0 = (y[sxy*iy] + yoff)*scale;
    y02 = y0*y0;

      /* Compute intermediaries. */
      x0 = (x[sxy*ix] + xoff)*scale;
      r2 = x0*x0 + y02;

      if (boff == 0.0) {
        /* Orthographic projection. */
        if (r2 != 0.0) {
          phi = atan2(x0, -y0);
        } else {
          phi = 0.0;
        }

        if (r2 < 0.5) {
          theta = acos(sqrt(r2));
        } else if (r2 <= 1.0) {
          theta = asin(sqrt(1.0 - r2));
        } else {
          stat[ix*spt+mx*iy*spt] = 1;
          if (!status) status = PRJERR_BAD_PIX_SET_SIN;
          return;
        }

      } else {
        /* "Synthesis" projection. */
        xy = x0*xi + y0*eta;

        if (r2 < 1.0e-10) {
          /* Use small angle formula. */
          z = r2/2.0;
          theta = 90.0 - R2D*sqrt(r2/(1.0 + xy));

        } else {
          a = aoff;
          b = xy + boff;
          c = r2 - xy - xy + coff;
          d = b*b - a*c;

          /* Check for a solution. */
          if (d < 0.0) {
            phi = 0.0;
            theta = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }
          d = sqrt(d);

          /* Choose solution closest to pole. */
          sinth1 = (-b + d)/a;
          sinth2 = (-b - d)/a;
          sinthe = (sinth1 > sinth2) ? sinth1 : sinth2;
          if (sinthe > 1.0) {
            if (sinthe-1.0 < tol) {
              sinthe = 1.0;
            } else {
              sinthe = (sinth1 < sinth2) ? sinth1 : sinth2;
            }
          }

          if (sinthe < -1.0) {
            if (sinthe+1.0 > -tol) {
              sinthe = -1.0;
            }
          }

          if (sinthe > 1.0 || sinthe < -1.0) {
            phi = 0.0;
            theta = 0.0;
            stat[ix*spt+mx*iy*spt] = 1;
            if (!status) status = PRJERR_BAD_PIX_SET_SIN;
            return;
          }

          theta = 0.0;
           asin(sinthe);
          z = 1.0 - sinthe;
        }

        x1 = -y0 + eta*z;
        y1 =  x0 -  xi*z;
        if (x1 == 0.0 && y1 == 0.0) {
          phi = 0.0;
        } else {
          phi = atan2(y1,x1);
        }
      }
  int mphi, mtheta;
  DATATYPE cosphi, costhe, sinphi, r, t, z1, z2;
  register int iphi, itheta, istat;

  mphi   = nx;
  mtheta = ny;

  /* Do theta dependence. */
  itheta = threadIdx.y + blockIdx.y * blockDim.y;
  iphi = threadIdx.x + blockIdx.x * blockDim.x;
  if (iphi>mphi || itheta>mtheta) return;
  //for (itheta = 0; itheta < ntheta; itheta++) {
    t = (90.0 - fabs(theta))*D2R;
    if (t < 1.0e-5) {
      if (theta > 0.0) {
         z = t*t/2.0;
      } else {
         z = 2.0 - t*t/2.0;
      }
      costhe = t;
    } else {
      z = 1.0 - sin(theta);
      costhe = cos(theta);
    }
    r = r0*costhe;

    if (scale == 0.0) {
      /* Orthographic projection. */
      istat = 0;
      if (bounds&1) {
        if (theta < 0.0) {
          istat = 1;
          if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
        }
      }

      //for (iphi = 0; iphi < mphi; iphi++, xp += sxy, yp += sxy) {
        sincos(phi, &sinphi, &cosphi); 
        x[iphi*sxy+mphi*sxy*itheta] =  r*sinphi - xoff_out;
        y[iphi*sxy+mphi*sxy*itheta] =  -r*cosphi - yoff_out;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}

    } else {
      /* "Synthesis" projection. */
      z *= r0;
      z1 = sintheta0*z - xoff_out;
      z2 = costheta0*z - yoff_out;

      //for (iphi = 0; iphi < mphi; iphi++) {
        istat = 0;
        sincos(phi, &sinphi, &cosphi); 
        if (bounds&1) {
          t = -atan(sintheta0*(sinphi) - costheta0*(cosphi));
          if (theta < t) {
            istat = 1;
            if (!status) status = PRJERR_BAD_WORLD_SET_SIN;
          }
        }

        x[iphi*sxy+mphi*sxy*itheta] = r*sinphi + z1;
        y[iphi*sxy+mphi*sxy*itheta] = -r*cosphi + z2;
        stat[iphi*sxy+mphi*sxy*itheta] = istat;
      //}
    }
  //}

   return;
}
__global__ void sinx2s_kernel(DATATYPE xi, DATATYPE eta, DATATYPE xoff, DATATYPE yoff, DATATYPE scale,
                DATATYPE aoff, DATATYPE boff, DATATYPE coff, int nx, int ny, int sxy, 
                int spt, DATATYPE *x, DATATYPE *y, DATATYPE *phi, DATATYPE *theta, 
                int *stat)                                                 {
  sinx2s_dev(xi, eta, xoff, yoff, scale, aoff, boff, coff, nx, ny, sxy, spt, 
             x, y, phi, theta, stat);
}
__global__ void sins2x_kernel(DATATYPE r0, DATATYPE scale, DATATYPE x0, DATATYPE y0, DATATYPE sintheta0,
               DATATYPE costheta0, int bounds, int nphi, int ntheta, int spt, 
               int sxy, DATATYPE *phi, DATATYPE *theta, DATATYPE *x, DATATYPE *y, int *stat) {
  sins2x_dev(r0, scale, x0, y0, sintheta0, costheta0, bounds, nphi, ntheta, sxy, spt, 
             phi, theta, x, y, stat);
}
__device__ void interp_dev(const DATATYPE* x3, const DATATYPE* y3, DATATYPE_INTERP2* img_orig, int sz, 
                              DATATYPE xgrid, DATATYPE ygrid, DATATYPE_INTERP2* img_out) {
      int z = threadIdx.x + blockIdx.x * blockDim.x;
      z += blockDim.x*gridDim.x*(threadIdx.y + blockIdx.y*blockDim.y); 
      if (z>=sz) return;
      DATATYPE thisx = x3[z]-IMGX0;
      DATATYPE thisy = y3[z]-IMGY0;
      int x0 = floorf(thisx/xgrid)+PAD_SIZE;
      DATATYPE xfrac = thisx/xgrid-x0+PAD_SIZE;
      int y0 = floorf(thisy/ygrid)+PAD_SIZE;
      DATATYPE yfrac = thisy/ygrid-y0+PAD_SIZE;
      int inx0 = IMG_PAD*y0+x0;
      inx0 %= IMG_PAD*IMG_PAD;
      DATATYPE_INTERP out_x = img_orig[inx0].x;
      DATATYPE_INTERP out_y = img_orig[inx0].y;
      out_x *= (1-xfrac)*(1-yfrac);
      out_y *= (1-xfrac)*(1-yfrac);
      out_x += (1-xfrac)*yfrac*img_orig[inx0+IMG_PAD].x;
      out_y += (1-xfrac)*yfrac*img_orig[inx0+IMG_PAD].y;
      out_x += xfrac*(1-yfrac)*img_orig[inx0+1].x;
      out_y += xfrac*(1-yfrac)*img_orig[inx0+1].y;
      out_x += xfrac*yfrac*img_orig[inx0+IMG_PAD+1].x;
      out_y += xfrac*yfrac*img_orig[inx0+IMG_PAD+1].y;
      img_out[z].x = out_x;
      img_out[z].y = out_y;
      //img_out[z].x = g00.x*(1-xfrac)*(1-yfrac)+g01.x*(1-xfrac)*yfrac+g10.x*xfrac*(1-yfrac)+g11.x*xfrac*yfrac;
      //img_out[z].y = g00.y*(1-xfrac)*(1-yfrac)+g01.y*(1-xfrac)*yfrac+g10.y*xfrac*(1-yfrac)+g11.y*xfrac*yfrac;
     
}
__global__ void interp_kernel(const DATATYPE* x3, const DATATYPE* y3, DATATYPE_INTERP2* img_orig, int sz, 
                              DATATYPE xgrid, DATATYPE ygrid, DATATYPE_INTERP2* img_out) {
  interp_dev(x3, y3, img_orig, sz, xgrid, ygrid, img_out);
}
__global__ void coord_convert(DATATYPE xi, DATATYPE eta, DATATYPE xoff, DATATYPE yoff, DATATYPE scale_in,
               DATATYPE aoff, DATATYPE boff, DATATYPE coff, int nx, int ny, 
               DATATYPE r0, DATATYPE scale_out, DATATYPE xoff_out, DATATYPE yoff_out, DATATYPE sintheta0,
               DATATYPE costheta0, int bounds, int nphi, int ntheta, 
               int sxy, int spt, DATATYPE *x, DATATYPE *y, DATATYPE *phi, DATATYPE *theta, 
#ifdef __USE_TEX
                cudaTextureObject_t img_orig, 
#else
                DATATYPE_INTERP2* img_orig,
#endif
               int sz, DATATYPE xgrid, DATATYPE ygrid, DATATYPE_INTERP2* img_out, 
               int *stat) {
  //coord_convert_dev(xi,eta, xoff, yoff, scale_in, aoff, boff, coff, nx, ny, sxy, spt, 
  //                  r0, scale_out, xoff_out, yoff_out, sintheta0, costheta0, bounds, 
  //                  x, y, stat);
  //sinx2s_dev(xi, eta, xoff, yoff, scale_in, aoff, boff, coff, nx, ny, sxy, spt, 
  //           x, y, phi, theta, stat);
  //sins2x_dev(r0, scale_out, xoff_out, yoff_out, sintheta0, costheta0, bounds, nphi, ntheta, sxy, spt, 
  //           phi, theta, x, y, stat);
  //cc_dev(xi, eta, xoff, yoff, scale_in, 
  //              aoff, boff, coff, nx, ny, nphi, ntheta, 
  //              sxy, spt, r0, scale_out, xoff_out, yoff_out,
  //              sintheta0, costheta0, bounds, x, y, stat,
  //              phi, theta);
  //interp_dev(x, y, img_orig, sz, xgrid, ygrid, img_out);
  reproject_dev(xi, eta, xoff, yoff, scale_in, 
                aoff, boff, coff, nx, ny, nphi, ntheta, 
                sxy, spt, r0, scale_out, xoff_out, yoff_out,
                sintheta0, costheta0, bounds, x, y, img_orig, sz,
                xgrid, ygrid, img_out, stat);
  
}


int main(void) {
   DATATYPE *x, *y, *phi, *theta, *x2, *y2;

   int *stat;

   struct prjprm prj;

   x = (DATATYPE*)malloc(sizeof(DATATYPE)*SIZE);
   y = (DATATYPE*)malloc(sizeof(DATATYPE)*SIZE);
   x2 = (DATATYPE*)malloc(sizeof(DATATYPE)*SIZE);
   y2 = (DATATYPE*)malloc(sizeof(DATATYPE)*SIZE);
   phi = (DATATYPE*)malloc(sizeof(DATATYPE)*SIZE);
   theta = (DATATYPE*)malloc(sizeof(DATATYPE)*SIZE);
   stat = (int*)malloc(sizeof(int)*SIZE);

   if (!x || !y || !x2 || !y2 || !phi || !theta || ! stat) {
      std::cout << "ERROR. CPU allocate failed." << std::endl;
   }

   /***   Initialize ***/
   srand(2541617);
   prj.flag = SIN;
   prj.pv[0] = (rand()*1.0)/RAND_MAX;
   prj.pv[1] = (rand()*1.0)/RAND_MAX;
   prj.pv[2] = (rand()*1.0)/RAND_MAX;
   prj.x0=0;
   prj.y0=0;
   prj.r0=0.663;
   prj.w[0] = (rand()*1.0)/RAND_MAX;
   prj.w[1] = 0.0;
   //prj.w[1] = (rand()*1.0)/RAND_MAX;
   prj.w[2] = (rand()*1.0)/RAND_MAX;
   prj.w[3] = (rand()*1.0)/RAND_MAX;
   prj.bounds = 0;

   for (int z=0;z<SIZE;z++) {
      x[z] = XLL + (z%SIZEX)*(XUR/SIZEX);
      y[z] = YLL + (z/SIZEX)*(YUR/SIZEY);
   }

   DATATYPE_INTERP2* img_orig;
   DATATYPE_INTERP2* img_out;
   DATATYPE_INTERP2* img_out2;
   img_orig = (DATATYPE_INTERP2*)malloc(sizeof(DATATYPE_INTERP2)*IMG_PAD*IMG_PAD);
   img_out = (DATATYPE_INTERP2*)malloc(sizeof(DATATYPE_INTERP2)*IMG_PAD*IMG_PAD);
   img_out2 = (DATATYPE_INTERP2*)malloc(sizeof(DATATYPE_INTERP2)*IMG_PAD*IMG_PAD);
   if (!img_orig || !img_out || !img_out2) std::cerr << "ERROR. Failed CPU alloc." <<std::endl;
   for (int z=0;z<IMG_PAD*IMG_PAD;z++) {
      img_orig[z].x = (rand()*1.0)/RAND_MAX;
      img_orig[z].y = (rand()*1.0)/RAND_MAX;
   }

   /*** Execute x2s two ways ***/
   if (sinx2s(&prj, SIZEX, SIZEY, 1, 1, x, y, phi, theta, stat)) 
                        std::cout << "ERROR in sinx2s" << std::endl;

   /*** Execute s2x two ways ***/
   if (sins2x_alt2(&prj, SIZEX, SIZEY, 1, 1, phi, theta, x2, y2, stat)) 
                        std::cout << "ERROR in sins2x" << std::endl;

   DATATYPE xmin, xmax, ymin, ymax;
   xmin = ymin = 100000000;
   xmax = ymax = -100000000;
   for (int z=0;z<SIZE;z++) {
      if (x2[z]<xmin) xmin = x2[z]; 
      if (y2[z]<ymin) ymin = y2[z]; 
      if (x2[z]>xmax) xmax = x2[z]; 
      if (y2[z]>ymax) ymax = y2[z]; 
   }
   std::cout << xmin << ", " << ymin << " -- " << xmax << ", " << ymax << std::endl;
   DATATYPE xgrid = (IMGX1-IMGX0)/IMG_SIZE;
   DATATYPE ygrid = (IMGY1-IMGY0)/IMG_SIZE;
   for (int z=0;z<SIZE;z++) {
      DATATYPE thisx = x2[z]-IMGX0;
      DATATYPE thisy = y2[z]-IMGY0;
      int x0 = floorf(thisx/xgrid)+PAD_SIZE;
      DATATYPE xfrac = thisx/xgrid-x0+PAD_SIZE;
      int y0 = floorf(thisy/ygrid)+PAD_SIZE;
      DATATYPE yfrac = thisy/ygrid-y0+PAD_SIZE;
      int inx0 = IMG_PAD*y0+x0;
#ifdef __INTERP_LINEAR
      DATATYPE_INTERP2 g00 = img_orig[inx0];
      DATATYPE_INTERP2 g01 = img_orig[inx0+IMG_PAD];
      DATATYPE_INTERP2 g10 = img_orig[inx0+1];
      DATATYPE_INTERP2 g11 = img_orig[inx0+IMG_PAD+1];
      img_out[z].x = g00.x*(1-xfrac)*(1-yfrac)+g01.x*(1-xfrac)*yfrac+g10.x*xfrac*(1-yfrac)+g11.x*xfrac*yfrac;
      img_out[z].y = g00.y*(1-xfrac)*(1-yfrac)+g01.y*(1-xfrac)*yfrac+g10.y*xfrac*(1-yfrac)+g11.y*xfrac*yfrac;
#else
      DATATYPE w[4];
      w[0] = (-xfrac*xfrac*xfrac + 3*xfrac*xfrac - 3*xfrac + 1)/6.0;
      w[1] = (3*xfrac*xfrac*xfrac - 6*xfrac*xfrac + 4)/6.0;
      w[2] = (-3*xfrac*xfrac*xfrac + 3*xfrac*xfrac + 3*xfrac + 1)/6.0;
      w[3] = xfrac*xfrac*xfrac/6.0;
      DATATYPE_INTERP2 ivals[4];
      inx0 -= IMG_PAD;
      for (int qq=0;qq<4;qq++,inx0+=IMG_PAD) {
         ivals[qq] = img_orig[inx0-1]*w[0]
                   + img_orig[inx0]*w[1]
                   + img_orig[inx0+1]*w[2]
                   + img_orig[inx0+2]*w[3];
      }
      img_out[z] = ivals[0]*(-yfrac*yfrac*yfrac+3*yfrac*yfrac-3*yfrac+1) +
                   ivals[1]*(3*yfrac*yfrac*yfrac-6*yfrac*yfrac-4) +
                   ivals[2]*(-3*yfrac*yfrac*yfrac+3*yfrac*yfrac+3*yfrac+1) +
                   ivals[3]*yfrac*yfrac*yfrac;
      img_out[z] = img_out[z] *(1.0/6.0);
   
#endif
   }

   /*** GPU memory ***/
   DATATYPE_INTERP2 *dimg_orig, *dimg_out;
   cudaMalloc(&dimg_orig, sizeof(DATATYPE_INTERP2)*IMG_PAD*IMG_PAD);
   cudaMalloc(&dimg_out, sizeof(DATATYPE_INTERP2)*IMG_PAD*IMG_PAD);
   if (!dimg_orig || !dimg_out) std::cerr << "ERROR: Failed GPU allocation." << std::endl;
   cudaMemcpy(dimg_orig, img_orig, sizeof(DATATYPE_INTERP2)*IMG_PAD*IMG_PAD, cudaMemcpyHostToDevice);
   checkCudaError(__LINE__,__FILE__);
#ifdef __USE_TEX
   cudaResourceDesc resDesc;
   memset(&resDesc, 0, sizeof(resDesc));
   resDesc.resType = cudaResourceTypeLinear;
   resDesc.res.linear.devPtr = (float*)dimg_orig;
   resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
   resDesc.res.linear.desc.x = 32;
   resDesc.res.linear.sizeInBytes = IMG_PAD*IMG_PAD*sizeof(DATATYPE_INTERP2);

   cudaTextureDesc texDesc;
   memset(&texDesc, 0, sizeof(texDesc));
   texDesc.readMode = cudaReadModeElementType;
   cudaTextureObject_t img_tex=0;
   cudaCreateTextureObject(&img_tex, &resDesc, &texDesc, NULL);
#endif

   DATATYPE *dx, *dy, *dphi, *dtheta;
   int *dstat;
   cudaMalloc(&dx, sizeof(DATATYPE)*SIZE);
   cudaMalloc(&dy, sizeof(DATATYPE)*SIZE);
   cudaMalloc(&dphi, sizeof(DATATYPE)*SIZE);
   cudaMalloc(&dtheta, sizeof(DATATYPE)*SIZE);
   cudaMalloc(&dstat, sizeof(int)*SIZE);

   
   cudaMemcpy(dx, x, sizeof(DATATYPE)*SIZE, cudaMemcpyHostToDevice);
   cudaMemcpy(dy, y, sizeof(DATATYPE)*SIZE, cudaMemcpyHostToDevice);
   checkCudaError(__LINE__,__FILE__);

   /*** Compute on GPU ***/
   cudaEvent_t start, finish;
   cudaEventCreate(&start); cudaEventCreate(&finish);
   cudaEventRecord(start);
   float elapsed;
   coord_convert<<<dim3((SIZEX+512-1)/512,SIZEY),512>>>(prj.pv[1], prj.pv[2], prj.x0, prj.y0, prj.w[0], 
                     prj.w[2], -prj.w[1], prj.w[3], SIZEX, SIZEY, prj.r0, prj.w[1], 
                     prj.x0, prj.y0, prj.pv[1], prj.pv[2], prj.bounds, SIZEX, SIZEY, 
                     1, 1, dx, dy, dphi, dtheta, 
#ifdef __USE_TEX
                     img_tex, 
#else
                     dimg_orig,
#endif
                     IMG_SIZE*IMG_SIZE, xgrid, ygrid,
                     dimg_out, dstat);
   cudaEventRecord(finish);
   cudaEventSynchronize(finish);
   cudaEventElapsedTime(&elapsed, start, finish);
   cudaEventDestroy(start); cudaEventDestroy(finish);
   checkCudaError(__LINE__,__FILE__);
   cudaMemcpy(img_out2, dimg_out, sizeof(DATATYPE_INTERP2)*IMG_PAD*IMG_PAD, cudaMemcpyDeviceToHost);
   checkCudaError(__LINE__,__FILE__);

   std::cout << "Elapsed time: " << elapsed << " ms." << std::endl;
   std::cout << "Throughput: " << 1.0/elapsed*SIZEX*SIZEY/1000/1000 << " Gpixels/sec." << std::endl;

   std::cout << "Check results against CPU..." << std::endl;

   double maxerr = 0.0;
   int maxerrinx = -1;
   for (int z=0;z<SIZE;z++) {
      if (0==z%1000 && 
          (fabs(img_out2[z].x-img_out[z].x) > 0.0001 ||
           fabs(img_out2[z].y-img_out[z].y) > 0.0001  ) ) {
         std::cout << "Mismatch for z = " << z << ": " << img_out2[z].x << ", " <<img_out2[z].y << " != "
                   << img_out[z].x << ", " << img_out[z].y << std::endl;
      }
      if (fabs(img_out2[z].x-img_out[z].x > maxerr)) {maxerr = fabs(img_out2[z].x-img_out[z].x); maxerrinx=z;}
      if (fabs(img_out2[z].y-img_out[z].y > maxerr)) {maxerr = fabs(img_out2[z].y-img_out[z].y); maxerrinx=z;}
   }
   std::cout << "Largest error at index " << maxerrinx << ".  " << img_out2[maxerrinx].x << ", " 
             << img_out2[maxerrinx].y << " != " << img_out[maxerrinx].x << ", "
             << img_out[maxerrinx].y << std::endl;
   free(x);
   free(y);
   free(x2);
   free(y2);
   free(phi);
   free(theta);
   free(stat);
   free(img_orig);
   free(img_out);
   free(img_out2);

#ifdef __USE_TEX
   cudaDestroyTextureObject(img_tex);
#endif
   cudaFree(dx);
   cudaFree(dy);
   cudaFree(dphi);
   cudaFree(dtheta);
   cudaFree(dstat);
   cudaFree(dimg_orig);
   cudaFree(dimg_out);

   
   return 0;
}
