from __future__ import division
from qm.model.model import Model
import numpy as np

class NS2D(Model):
    """ Class for 2-dimensional non-separable model BO calculation

        :param object molecule: molecule object
        :param double A: parameter for 2-dimensional non-separable model 
        :param double B: parameter for 2-dimensional non-separable model 
        :param double C: parameter for 2-dimensional non-separable model 
        :param double D: parameter for 2-dimensional non-separable model 
        :param double D: parameter for 2-dimensional non-separable model 
    """
    def __init__(self, molecule, A=0.15, B=0.14, C=0.015, D=0.06, E=0.05):
        # Initialize model common variables
        super(NS2D, self).__init__(None)

        # Define parameters
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E

        # Set 'l_nacme' with respect to the computational method
        # DAG model can produce NACs, so we do not need to get NACME
        molecule.l_nacme = False

        # DAG model can compute the gradient of several states simultaneously
        self.re_calc = False

    def get_data(self, molecule, base_dir, bo_list, dt, istep, calc_force_only, md=None, itraj=None):
        """ Extract energy, gradient and nonadiabatic couplings from double arch geometry model BO calculation

            :param object molecule: molecule object
            :param string base_dir: base directory
            :param integer,list bo_list: list of BO states for BO calculation
            :param double dt: time interval
            :param integer istep: current MD step
            :param boolean calc_force_only: logical to decide whether calculate force only
        """
        # Initialize diabatic Hamiltonian
        H = np.zeros((2, 2))
        dH = np.zeros((2, 2, 2))
        unitary = np.zeros((2, 2))

        x = molecule.pos[0, 0]
        y = molecule.pos[0, 1]

        # Define Hamiltonian
        H[0, 0] = - self.E
        H[1, 1] = - self.A * np.exp(- self.B * (0.75 * (x + y) ** 2 + 0.25 * (x - y) ** 2))
        H[0, 1] =   self.C * np.exp(- self.D * (0.25 * (x + y) ** 2 + 0.75 * (x - y) ** 2))
        H[1, 0] =  H[0, 1] 

        # Define a derivative of Hamiltonian
        dH[0, 0, :] = 0.
        dH[1, 1, 0] = - self.B * H[1, 1] * (1.5 * (x + y) + 0.5 * (x - y))
        dH[1, 1, 1] = - self.B * H[1, 1] * (1.5 * (x + y) - 0.5 * (x - y))
        
        dH[0, 1, 0] = - self.D * H[0, 1] * (0.5 * (x + y) + 1.5 * (x - y))
        dH[0, 1, 1] = - self.D * H[0, 1] * (0.5 * (x + y) - 1.5 * (x - y))
        dH[1, 0, :] = dH[0, 1, :] 

        # Diagonalization
        a = 4. * H[1, 0] * H[0, 1] + (H[1, 1] - H[0, 0]) ** 2
        sqa = np.sqrt(a)
        tantheta = (H[1, 1] - H[0, 0] - sqa) / H[1, 0]  * 0.5
        theta = np.arctan(tantheta)

        unitary[0, 0] = np.cos(theta)
        unitary[1, 0] = np.sin(theta)
        unitary[0, 1] = - np.sin(theta)
        unitary[1, 1] = np.cos(theta)

        # Extract adiabatic quantities
        molecule.states[0].energy = 0.5 * (H[0, 0] + H[1, 1]) - 0.5 * sqa
        molecule.states[1].energy = 0.5 * (H[0, 0] + H[1, 1]) + 0.5 * sqa

        #molecule.states[0].force = np.dot(unitary[:, 1], np.matmul(dH, unitary[:, 1]))
        #molecule.states[1].force = np.dot(unitary[:, 0], np.matmul(dH, unitary[:, 0]))
        molecule.states[0].force[0, :] = - np.einsum('i,ija,j->a', unitary[:, 0], dH[:, :, :], unitary[:, 0])
        molecule.states[1].force[0, :] = - np.einsum('i,ija,j->a', unitary[:, 1], dH[:, :, :], unitary[:, 1])

        molecule.nac[0, 1, 0, :] = np.einsum('i,ija,j->a', unitary[:, 0], dH[:, :, :], unitary[:, 1]) / sqa 
        molecule.nac[1, 0, 0, :] = - molecule.nac[0, 1, 0, :]

        if (md is not None):
            if (md.l_lap):
                
                d2H = np.zeros((2, 2))
                d2H[0, 0] = 0.
                d2H[1, 1] = 0.

                if (abs(x) > self.D):
                    d2H[1, 0] = np.sign(x) * self.B * self.C ** 2 * np.exp(- np.sign(x) * self.C * (x - self.D)) \
                             - np.sign(x) * self.B * self.C ** 2 * np.exp(- np.sign(x) * self.C * (x + self.D))
                else:
                    d2H[1, 0] = - self.B * self.C ** 2 * np.exp(self.C * (x - self.D)) \
                                - self.B * self.C ** 2 * np.exp(- self.C * (x + self.D))

                dH[0, 1] = dH[1, 0]
                
                if (H[0, 1] > 1e-8):
                    dtheta = ((- molecule.states[0].force - dH[0, 0])* H[0, 1]  - dH[0, 1] * (molecule.states[0].energy - H[0, 0])) / H[0, 1] ** 2
                else:
                    dtheta = 0.0

                md.d2e[itraj, 0, 0] = 2.0 * dtheta * dH[0, 1] * (np.sin(theta) ** 2 - np.cos(theta) ** 2) \
                    + np.dot(unitary[:, 0], np.matmul(d2H, unitary[:, 0]))
                md.d2e[itraj, 1, 0] = - 2.0 * dtheta * dH[0, 1] * (np.sin(theta) ** 2 - np.cos(theta) ** 2) \
                    + np.dot(unitary[:, 1], np.matmul(d2H, unitary[:, 1]))

