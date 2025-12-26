#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

static void xfpc_cdot(int nst, int *l_coh, double **epc, double complex *c, double complex *xfpccdot){
    
    int ist, jst;
    for(ist = 0; ist < nst; ist++){
        xfpccdot[ist] = 0.0 + 0.0 * I;
        if(l_coh[ist] == 1){
            for(jst = 0; jst < nst; jst++){
                xfpccdot[ist] += - 1.0 * I * c[ist] * (epc[ist][jst] - epc[0][jst]) * creal(conj(c[jst]) * c[jst]);
            }
        }
    }

}

// Routine to calculate cdot contribution originated from XF term
static void xf_cdot(int nat, int ndim, int nst, int *l_coh, int l_crunch, double *mass, double *sigma,
    double **pos, double ***aux_pos, double ***phase, double ****qmom, double complex *c, double complex *xfcdot){

    double **avg_pos = malloc(nat * sizeof(double*));
    double **dec = malloc(nst * sizeof(double*));
    double *rho = malloc(nst * sizeof(double));

    int ist, jst, iat, isp;

    // Initialize variables related to decoherence
    for(ist = 0; ist < nst; ist++){
        for(jst = 0; jst < nst; jst++){
            for(iat = 0; iat < nat; iat++){
                for(isp = 0; isp < ndim; isp++){
                    qmom[ist][jst][iat][isp] = 0.0;
                }
            }
        }
    }

    for(iat = 0; iat < nat; iat++){
        avg_pos[iat] = malloc(ndim * sizeof(double));
        for(isp = 0; isp < ndim; isp++){
            avg_pos[iat][isp] = 0.0;
        }
    }
    
    for(ist = 0; ist < nst; ist++){
        dec[ist] = malloc(nst * sizeof(double));
        for(jst = 0; jst < nst; jst++){
            dec[ist][jst] = 0.0;
        }
    }

    // Calculate densities from current coefficients
    for(ist = 0; ist < nst; ist++){
        rho[ist] = creal(conj(c[ist]) * c[ist]);
        if(l_coh[ist] == 1){
            for(iat = 0; iat < nat; iat++){
                for(isp = 0; isp < ndim; isp++){
                    avg_pos[iat][isp] += aux_pos[ist][iat][isp] * rho[ist];
                }
            }
        }
    }

    // Get quantum momentum from auxiliary positions and sigma values
    if(l_crunch == 1){
        for(ist = 0; ist < nst; ist++){
            for(jst = ist + 1; jst < nst; jst++){
                if(l_coh[ist] == 1 && l_coh[jst] == 1){
                    for(iat = 0; iat < nat; iat++){
                        for(isp = 0; isp < ndim; isp++){
                            qmom[ist][jst][iat][isp] = 
                                - (pos[iat][isp] - aux_pos[ist][iat][isp] - aux_pos[jst][iat][isp] + avg_pos[iat][isp])
                                / pow(sigma[iat], 2.0) / mass[iat];
                            qmom[jst][ist][iat][isp] = qmom[ist][jst][iat][isp];
                        }
                    }
                }
            }
        }
    }
    else{
        for(ist = 0; ist < nst; ist++){
            for(jst = ist + 1; jst < nst; jst++){
                if(l_coh[ist] == 1 && l_coh[jst] == 1){
                    for(iat = 0; iat < nat; iat++){
                        for(isp = 0; isp < ndim; isp++){
                            qmom[ist][jst][iat][isp] = - (pos[iat][isp] - avg_pos[iat][isp])
                                / pow(sigma[iat], 2.0) / mass[iat];
                            qmom[jst][ist][iat][isp] = - (pos[iat][isp] - avg_pos[iat][isp])
                                / pow(sigma[iat], 2.0) / mass[iat];
                        }
                    }
                }
            }
        }
    }

    // Get decoherence term from quantum momentum and phase
    for(ist = 0; ist < nst; ist++){
        for(jst = ist + 1; jst < nst; jst++){
            if(l_coh[ist] == 1 && l_coh[jst] == 1){
                for(iat = 0; iat < nat; iat++){
                    for(isp = 0; isp < ndim; isp++){
                        dec[ist][jst] += qmom[ist][jst][iat][isp] * (phase[ist][iat][isp] - phase[jst][iat][isp]);
                    }
                }
            }
            dec[jst][ist] = - 1.0 * dec[ist][jst];

        }
    }

    // Get cdot contribution from decoherence term
    for(ist = 0; ist < nst; ist++){
        xfcdot[ist] = 0.0 + 0.0 * I;
        for(jst = 0; jst < nst; jst++){
            xfcdot[ist] -= 0.5 * dec[ist][jst] * rho[jst] * c[ist];
        }
    }

    // Deallocate temporary arrays
    for(ist = 0; ist < nst; ist++){
        free(dec[ist]);
    }
    for(iat = 0; iat < nat; iat++){
        free(avg_pos[iat]);
    }
    free(dec);
    free(rho);
    free(avg_pos);

}

// Routine to print xf debug info 
static void xf_print_coef(int nst, double complex *coef, double complex *xfcdot, double *dotpopdec){
    int ist;
    
    for(ist = 0; ist < nst; ist++){
        dotpopdec[ist] = 2.0 * creal(xfcdot[ist] * conj(coef[ist]));
    }
}

// Routine to calculate rhodot contribution originated from XF term
static void xf_rhodot(int nat, int ndim, int nst, int *l_coh, double *mass, double *sigma,
    double **pos, double **qmom, double ***aux_pos, double ***phase, double complex **rho, double complex **xfrhodot){

    double **dec = malloc(nst * sizeof(double*));

    int ist, jst, kst, iat, isp;

    // Initialize variables related to decoherence
    for(iat = 0; iat < nat; iat++){
        for(isp = 0; isp < ndim; isp++){
            qmom[iat][isp] = 0.0;
        }
    }

    for(ist = 0; ist < nst; ist++){
        dec[ist] = malloc(nst * sizeof(double));
        for(jst = 0; jst < nst; jst++){
            dec[ist][jst] = 0.0;
        }
    }

    // Get quantum momentum from auxiliary positions and sigma values
    for(ist = 0; ist < nst; ist++){

        if(l_coh[ist] == 1){
            for(iat = 0; iat < nat; iat++){
                for(isp = 0; isp < ndim; isp++){
                    qmom[iat][isp] += 0.5 * creal(rho[ist][ist]) * (pos[iat][isp] - aux_pos[ist][iat][isp])
                        / pow(sigma[iat], 2.0) / mass[iat];
                }
            }
        }

    }

    // Get decoherence term from quantum momentum and phase
    for(ist = 0; ist < nst; ist++){
        for(jst = ist + 1; jst < nst; jst++){

            if(l_coh[ist] == 1 && l_coh[jst] == 1){
                for(iat = 0; iat < nat; iat++){
                    for(isp = 0; isp < ndim; isp++){
                        dec[ist][jst] += qmom[iat][isp] * (phase[ist][iat][isp] - phase[jst][iat][isp]);
                    }
                }
            }
            dec[jst][ist] = - 1.0 * dec[ist][jst];

        }
    }

    // Get rhodot contribution from decoherence term
    for(ist = 0; ist < nst; ist++){
        // Diagonal components
        xfrhodot[ist][ist] = 0.0 + 0.0 * I;
        for(kst = 0; kst < nst; kst++){
            xfrhodot[ist][ist] -= 2.0 * dec[kst][ist] * rho[ist][kst] * rho[kst][ist];
        }
        // Off-diagonal components
        for(jst = ist + 1; jst < nst; jst++){
            xfrhodot[ist][jst] = 0.0 + 0.0 * I;
            for(kst = 0; kst < nst; kst++){
                xfrhodot[ist][jst] -= (dec[kst][ist] + dec[kst][jst]) * rho[ist][kst] * rho[kst][jst];
            }
            xfrhodot[jst][ist] = conj(xfrhodot[ist][jst]);
        }
    }

    // Deallocate temporary arrays
    for(ist = 0; ist < nst; ist++){
        free(dec[ist]);
    }

    free(dec);

}

// Routine to print xf debug info 
static void xf_print_rho(int nst, double complex **xfrhodot, double *dotpopdec){
    int ist;
    
    for(ist = 0; ist < nst; ist++){
        dotpopdec[ist] = creal(xfrhodot[ist][ist]);
    }
}

