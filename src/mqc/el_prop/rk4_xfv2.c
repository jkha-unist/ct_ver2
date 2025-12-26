#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "derivs.h"
#include "derivs_xfv2.h"

// Routine for coefficient propagation scheme in rk4 propagator
static void rk4_coef(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh, int l_crunch, int t_pc,
    double *mass, double *energy, double *energy_old, double *sigma, double **nacme, double **nacme_old, 
    double **pos, double **pos_old, double ***aux_pos, double ***aux_pos_old,
    double ***phase, double ***phase_old, double **pc_term, double **pc_term_old, 
    double ****qmom, double complex *coef, int verbosity, double *dotpopdec);

// Routine for density propagation scheme in rk4 propagator
//static void rk4_rho(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
//    double *mass, double *energy, double *energy_old, double *sigma, double **nacme,
//    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex **rho,
//    int verbosity, double *dotpopdec);

// Interface routine for propagation scheme in rk4 propagator
static void rk4(int nat, int ndim, int nst, int nesteps, double dt, char *elec_object, int *l_coh, int l_crunch, int t_pc,
    double *mass, double *energy, double *energy_old, double *sigma, double **nacme, double **nacme_old,
    double **pos, double **pos_old, double ***aux_pos, double ***aux_pos_old, double ***phase, double ***phase_old,
    double **pc_term, double **pc_term_old, double ****qmom, double complex *coef, double complex **rho,
    int verbosity, double *dotpopdec){

    if(strcmp(elec_object, "coefficient") == 0){
        rk4_coef(nat, ndim, nst, nesteps, dt, l_coh, l_crunch, t_pc, mass, energy, energy_old, sigma,
            nacme, nacme_old, pos, pos_old, aux_pos, aux_pos_old, phase, phase_old, pc_term, pc_term_old, qmom, coef, verbosity, dotpopdec);
    }
//    else if(strcmp(elec_object, "density") == 0){
//        rk4_rho(nat, ndim, nst, nesteps, dt, l_coh, mass, energy, energy_old, sigma,
//            nacme, nacme_old, pos, qmom, aux_pos, phase, rho, verbosity, dotpopdec);
//    }

}

// Routine for coefficient propagation scheme in rk4 propagator
static void rk4_coef(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh, int l_crunch, int t_pc,
    double *mass, double *energy, double *energy_old, double *sigma, double **nacme, double **nacme_old,
    double **pos, double **pos_old, double ***aux_pos, double ***aux_pos_old, 
    double ***phase, double ***phase_old, double **pc_term, double **pc_term_old,
    double ****qmom, double complex *coef, int verbosity, double *dotpopdec){

    double complex *k1 = malloc(nst * sizeof(double complex));
    double complex *k2 = malloc(nst * sizeof(double complex));
    double complex *k3 = malloc(nst * sizeof(double complex));
    double complex *k4 = malloc(nst * sizeof(double complex));
    double complex *kfunction = malloc(nst * sizeof(double complex));
    double complex *variation = malloc(nst * sizeof(double complex));
    double complex *c_dot = malloc(nst * sizeof(double complex));
    double complex *xf_c_dot = malloc(nst * sizeof(double complex));
    double complex *xfpc_c_dot = malloc(nst * sizeof(double complex));
    double complex *coef_new = malloc(nst * sizeof(double complex));
    double *eenergy = malloc(nst * sizeof(double));
    double **dv = malloc(nst * sizeof(double*));
    double **epos = malloc(nat * sizeof(double*));
    double ***eaux_pos = malloc(nst * sizeof(double***));
    double ***ephase = malloc(nst * sizeof(double***));
    double **epc = malloc(nst * sizeof(double*));
    int ist, jst, iat, idim, iestep;
    double frac, edt, norm;
    

    for(ist = 0; ist < nst; ist++){
        dv[ist] = malloc(nst * sizeof(double));
        epc[ist] = malloc(nst * sizeof(double));
        eaux_pos[ist] = malloc(nat * sizeof(double));
        ephase[ist] = malloc(nat * sizeof(double));
        for(iat = 0; iat < nat; iat++){
            eaux_pos[ist][iat] = malloc(ndim * sizeof(double));
            ephase[ist][iat] = malloc(ndim * sizeof(double));
        }
        xf_c_dot[ist] = 0.0 + 0.0 * I;
        xfpc_c_dot[ist] = 0.0 + 0.0 * I;
    }
    
    for(iat = 0; iat < nat; iat++){
        epos[iat] = malloc(ndim * sizeof(double));
    }

    frac = 1.0 / (double)nesteps;
    edt = dt * frac;
    

    for(iestep = 0; iestep < nesteps; iestep++){

        // Interpolate energy and NACME terms between time t and t + dt
        for(ist = 0; ist < nst; ist++){
            eenergy[ist] = energy_old[ist] + (energy[ist] - energy_old[ist]) * (double)iestep * frac;
            for(jst = 0; jst < nst; jst++){
                dv[ist][jst] = nacme_old[ist][jst] + (nacme[ist][jst] - nacme_old[ist][jst])
                    * (double)iestep * frac;
            }
            for(iat = 0; iat < nat; iat++){
                for(idim = 0; idim < ndim; idim++){
                    eaux_pos[ist][iat][idim] = aux_pos_old[ist][iat][idim] + (aux_pos[ist][iat][idim] - aux_pos_old[ist][iat][idim]) 
                        * (double)iestep * frac;
                    ephase[ist][iat][idim] = phase_old[ist][iat][idim] + (phase[ist][iat][idim] - phase_old[ist][iat][idim]) 
                        * (double)iestep * frac;
                }
            }
        }

        // Interpolate phase correction term for t_pc == 2 case
        if(t_pc == 2){
            for(ist = 0; ist < nst; ist++){
                for(jst = 0; jst < nst; jst++){
                    epc[ist][jst] = pc_term_old[ist][jst] + (pc_term[ist][jst] - pc_term_old[ist][jst])
                        * (double)iestep * frac;
                }
            }
        }

        for(iat = 0; iat < nat; iat++){
            for(idim = 0; idim < ndim; idim++){
                epos[iat][idim] = pos_old[iat][idim] + (pos[iat][idim] - pos_old[iat][idim]) * (double)iestep * frac;
            }
        }

        // Calculate k1
        cdot(nst, eenergy, dv, coef, c_dot);
        xf_cdot(nat, ndim, nst, l_coh, l_crunch, mass, sigma, epos, eaux_pos, ephase, qmom, coef, xf_c_dot);
        if(t_pc == 2){
            xfpc_cdot(nst, l_coh, epc, coef, xfpc_c_dot);
        }

        for(ist = 0; ist < nst; ist++){
            k1[ist] = edt * (c_dot[ist] + xf_c_dot[ist] + xfpc_c_dot[ist]);
            kfunction[ist] = 0.5 * k1[ist];
            coef_new[ist] = coef[ist] + kfunction[ist];
        }

        // Calculate k2
        cdot(nst, eenergy, dv, coef_new, c_dot);
        xf_cdot(nat, ndim, nst, l_coh, l_crunch, mass, sigma, epos, eaux_pos, ephase, qmom, coef_new, xf_c_dot);
        if(t_pc == 2){
            xfpc_cdot(nst, l_coh, epc, coef_new, xfpc_c_dot);
        }
        
        for(ist = 0; ist < nst; ist++){
            k2[ist] = edt * (c_dot[ist] + xf_c_dot[ist] + xfpc_c_dot[ist]);
            kfunction[ist] = 0.5 * (- 1.0 + sqrt(2.0)) * k1[ist] + (1.0 - 0.5 * sqrt(2.0)) * k2[ist];
            coef_new[ist] = coef[ist] + kfunction[ist];
        }

        // Calculate k3
        cdot(nst, eenergy, dv, coef_new, c_dot);
        xf_cdot(nat, ndim, nst, l_coh, l_crunch, mass, sigma, epos, eaux_pos, ephase, qmom, coef_new, xf_c_dot);
        if(t_pc == 2){
            xfpc_cdot(nst, l_coh, epc, coef_new, xfpc_c_dot);
        }

        for(ist = 0; ist < nst; ist++){
            k3[ist] = edt * (c_dot[ist] + xf_c_dot[ist] + xfpc_c_dot[ist]);
            kfunction[ist] = - 0.5 * sqrt(2.0) * k2[ist] + (1.0 + 0.5 * sqrt(2.0)) * k3[ist];
            coef_new[ist] = coef[ist] + kfunction[ist];
        }

        // Calculate k4
        cdot(nst, eenergy, dv, coef_new, c_dot);
        xf_cdot(nat, ndim, nst, l_coh, l_crunch, mass, sigma, epos, eaux_pos, ephase, qmom, coef_new, xf_c_dot);
        if(t_pc == 2){
            xfpc_cdot(nst, l_coh, epc, coef_new, xfpc_c_dot);
        }

        for(ist = 0; ist < nst; ist++){
            k4[ist] = edt * (c_dot[ist] + xf_c_dot[ist] + xfpc_c_dot[ist]);
            variation[ist] = (k1[ist] + (2.0 - sqrt(2.0)) * k2[ist] + (2.0 + sqrt(2.0))
                * k3[ist] + k4[ist]) / 6.0;
            coef_new[ist] = coef[ist] + variation[ist];
        }

        // TODO : Is this part necessary?
        // Renormalize the coefficients
        norm = dot(nst, coef_new, coef_new);
        for(ist = 0; ist < nst; ist++){
            coef_new[ist] /= sqrt(norm);
            coef[ist] = coef_new[ist];
        }

    }

    if(verbosity >= 1){
        xf_print_coef(nst, coef, xf_c_dot, dotpopdec);
    }

    for(ist = 0; ist < nst; ist++){
        free(dv[ist]);
        free(epc[ist]);
    }

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(kfunction);
    free(variation);
    free(c_dot);
    free(xf_c_dot);
    free(xfpc_c_dot);
    free(coef_new);
    free(eenergy);
    free(dv);
    free(epos);
    free(eaux_pos);
    free(ephase);
    free(epc);

}

// Routine for density propagation scheme in rk4 propagator
//static void rk4_rho(int nat, int ndim, int nst, int nesteps, double dt, int *l_coh,
//    double *mass, double *energy, double *energy_old, double *sigma, double **nacme,
//    double **nacme_old, double **pos, double **qmom, double ***aux_pos, double ***phase, double complex **rho,
//    int verbosity, double *dotpopdec){
//
//    double complex **k1 = malloc(nst * sizeof(double complex*));
//    double complex **k2 = malloc(nst * sizeof(double complex*));
//    double complex **k3 = malloc(nst * sizeof(double complex*));
//    double complex **k4 = malloc(nst * sizeof(double complex*));
//    double complex **kfunction = malloc(nst * sizeof(double complex*));
//    double complex **variation = malloc(nst * sizeof(double complex*));
//    double complex **rho_dot = malloc(nst * sizeof(double complex*));
//    double complex **xf_rho_dot = malloc(nst * sizeof(double complex*));
//    double complex **rho_new = malloc(nst * sizeof(double complex*));
//    double *eenergy = malloc(nst * sizeof(double));
//    double **dv = malloc(nst * sizeof(double*));
//
//    int ist, jst, iestep;
//    double frac, edt;
//
//    for(ist = 0; ist < nst; ist++){
//        k1[ist] = malloc(nst * sizeof(double complex));
//        k2[ist] = malloc(nst * sizeof(double complex));
//        k3[ist] = malloc(nst * sizeof(double complex));
//        k4[ist] = malloc(nst * sizeof(double complex));
//        kfunction[ist] = malloc(nst * sizeof(double complex));
//        variation[ist] = malloc(nst * sizeof(double complex));
//        rho_dot[ist] = malloc(nst * sizeof(double complex));
//        xf_rho_dot[ist] = malloc(nst * sizeof(double complex));
//        rho_new[ist] = malloc(nst * sizeof(double complex));
//        dv[ist] = malloc(nst * sizeof(double));
//    }
//
//    frac = 1.0 / (double)nesteps;
//    edt = dt * frac;
//
//    for(iestep = 0; iestep < nesteps; iestep++){
//
//        // Calculate rhodot contribution originated from XF term
//        xf_rhodot(nat, ndim, nst, l_coh, mass, sigma, pos, qmom, aux_pos, phase, rho, xf_rho_dot);
//
//        // Interpolate energy and NACME terms between time t and t + dt
//        for(ist = 0; ist < nst; ist++){
//            eenergy[ist] = energy_old[ist] + (energy[ist] - energy_old[ist]) * (double)iestep * frac;
//            for(jst = 0; jst < nst; jst++){
//                dv[ist][jst] = nacme_old[ist][jst] + (nacme[ist][jst] - nacme_old[ist][jst])
//                    * (double)iestep * frac;
//            }
//        }
//
//        // Calculate k1
//        rhodot(nst, eenergy, dv, rho, rho_dot);
//
//        for(ist = 0; ist < nst; ist++){
//            for(jst = 0; jst < nst; jst++){
//                k1[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
//                kfunction[ist][jst] = 0.5 * k1[ist][jst];
//                rho_new[ist][jst] = rho[ist][jst] + kfunction[ist][jst];
//            }
//        }
//
//        // Calculate k2
//        rhodot(nst, eenergy, dv, rho_new, rho_dot);
//
//        for(ist = 0; ist < nst; ist++){
//            for(jst = 0; jst < nst; jst++){
//                k2[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
//                kfunction[ist][jst] = 0.5 * (- 1.0 + sqrt(2.0)) * k1[ist][jst]
//                    + (1.0 - 0.5 * sqrt(2.0)) * k2[ist][jst];
//                rho_new[ist][jst] = rho[ist][jst] + kfunction[ist][jst];
//            }
//        }
//
//        // Calculate k3
//        rhodot(nst, eenergy, dv, rho_new, rho_dot);
//
//        for(ist = 0; ist < nst; ist++){
//            for(jst = 0; jst < nst; jst++){
//                k3[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
//                kfunction[ist][jst] = - 0.5 * sqrt(2.0) * k2[ist][jst]
//                    + (1.0 + 0.5 * sqrt(2.0)) * k3[ist][jst];
//                rho_new[ist][jst] = rho[ist][jst] + kfunction[ist][jst];
//            }
//        }
//
//        // Calculate k4
//        rhodot(nst, eenergy, dv, rho_new, rho_dot);
//
//        for(ist = 0; ist < nst; ist++){
//            for(jst = 0; jst < nst; jst++){
//                k4[ist][jst] = edt * (rho_dot[ist][jst] + xf_rho_dot[ist][jst]);
//                variation[ist][jst] = (k1[ist][jst] + (2.0 - sqrt(2.0)) * k2[ist][jst]
//                    + (2.0 + sqrt(2.0)) * k3[ist][jst] + k4[ist][jst]) / 6.0;
//                rho[ist][jst] += variation[ist][jst];
//            }
//        }
//
//    }
//
//    if(verbosity >= 1){
//        xf_print_rho(nst, xf_rho_dot, dotpopdec); 
//    }
//
//    for(ist = 0; ist < nst; ist++){
//        free(k1[ist]);
//        free(k2[ist]);
//        free(k3[ist]);
//        free(k4[ist]);
//        free(kfunction[ist]);
//        free(variation[ist]);
//        free(rho_dot[ist]);
//        free(xf_rho_dot[ist]);
//        free(rho_new[ist]);
//        free(dv[ist]);
//    }
//
//    free(k1);
//    free(k2);
//    free(k3);
//    free(k4);
//    free(kfunction);
//    free(variation);
//    free(rho_dot);
//    free(xf_rho_dot);
//    free(rho_new);
//    free(eenergy);
//    free(dv);
//
//}
//
//
