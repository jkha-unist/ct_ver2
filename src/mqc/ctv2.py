from __future__ import division
from build.el_propagator_ctv2 import el_run
from mqc.mqc import MQC
from misc import eps, au_to_K, au_to_A, call_name, typewriter, gaussian1d, to_zero
import os, shutil, textwrap
import numpy as np
import pickle

class CTv2(MQC):
    """ Class for coupled-trajectory mixed quantum-classical (CTMQC) dynamics

        :param object,list molecules: List for molecule objects
        :param object thermostat: Thermostat object
        :param integer,list istates: List for initial state
        :param double dt: Time interval
        :param integer nsteps: Total step of nuclear propagation
        :param integer nesteps: Total step of electronic propagation
        :param string elec_object: Electronic equation of motions
        :param string propagator: Electronic propagator
        :param boolean l_print_dm: Logical to print BO population and coherence
        :param boolean l_adj_nac: Adjust nonadiabatic coupling to align the phases
        :param double rho_threshold: Electronic density threshold for decoherence term calculation
        :param init_coefs: Initial BO coefficient
        :type init_coefs: double, 2D list or complex, 2D list
        :param string unit_dt: Unit of time interval
        :param integer out_freq: Frequency of printing output
        :param integer verbosity: Verbosity of output
        :param boolean l_crunch: Perform CTv2
        :param boolean l_dc_w_mom: Use state-wise momentum for the phase term
        :param boolean l_traj_gaussian: Use the sum of trajectory centered Gaussians for the nuclear density
        :param integer t_cons: Average population conservation scheme. 0: none, 1: Scaling (only for l_lap=True), 2: Shift
        :param boolean l_etot0: Use the constant total energy (at t=0) for the state-wise momentum calculation
        :param boolean l_lap: Include laplacian ENC term
        :param boolean l_en_cons: Adjust momentum at every time step to enforce the total energy conservation
        :param double artifact_expon: Exponent for width for nuclear density estimation (used only when l_traj_gaussian = True)
        :param boolean l_asymp: Terminate dynamics when the trajectory reaches asymptotic region (use this option for model systems only)
        :param double x_fin: Define asymptotic region (a.u.)
        :param boolean l_real_pop: Use |C_j|^2 for |\chi_j|^2/|\chi|^2 in quantum momentum calculation.
        :param integer t_pc: Phase correction scheme (1: use P, 2: use \sum_j \nabla S_j)
    """
    def __init__(self, molecules, thermostat=None, istates=None, dt=0.5, nsteps=1000, nesteps=20, \
        elec_object="coefficient", propagator="rk4", l_print_dm=True, l_adj_nac=True, rho_threshold=0.01, \
        init_coefs=None, unit_dt="fs", out_freq=1, verbosity=0, \
        l_crunch=True, l_dc_w_mom=True, l_traj_gaussian=False, \
        t_cons=2, l_etot0=True, l_lap=False,\
        l_en_cons=False, artifact_expon=0.2, l_asymp=False, x_fin=25.0, \
        l_real_pop=True, t_pc=1):
        # Save name of MQC dynamics
        self.md_type = self.__class__.__name__

        # Initialize input values
        self.mols = molecules
        self.ntrajs = len(self.mols)
        self.digit = len(str(self.ntrajs))

        self.nst = self.mols[0].nst
        self.nat_qm = self.mols[0].nat_qm
        self.ndim = self.mols[0].ndim

        # Check compatibility between istates and init_coefs
        self.istates = istates
        self.init_coefs = init_coefs
        self.check_istates()

        # Initialize input values and coefficient for first trajectory
        super().__init__(self.mols[0], thermostat, self.istates[0], dt, nsteps, nesteps, \
            elec_object, propagator, l_print_dm, l_adj_nac, self.init_coefs[0], unit_dt, out_freq, verbosity)

        # Exception for electronic propagation
        if (self.elec_object != "coefficient"):
            error_message = "Electronic equation motion in CTMQC is only solved with respect to coefficient!"
            error_vars = f"elec_object = {self.elec_object}"
            raise NotImplementedError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Exception for thermostat
        if (self.thermo != None):
            error_message = "Thermostat is not implemented yet!"
            error_vars = f"thermostat = {self.thermo}"
            raise NotImplementedError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Initialize coefficient for other trajectories
        for itraj in range(1, self.ntrajs):
            self.mols[itraj].get_coefficient(self.init_coefs[itraj], self.istates[itraj])

        # Initialize variables for CTMQC
        self.phase = np.zeros((self.ntrajs, self.nst, self.nat_qm, self.ndim)) # phase term, same as self.dS if l_dc_w_mom=True
        self.nst_pair = int(self.nst * (self.nst - 1) / 2) # number of state pair
        self.qmom = np.zeros((self.ntrajs, self.nat_qm, self.ndim)) # total quantum momentum: \mathcal{P}_\nu = \nabla_\nu|\chi|^2 / |\chi|^2
        self.qmom_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim)) # projected quantum momentum: \mathcal{G}_{\nu,ij} =\nabla_\nu|\chi_i|^2 / |\chi_i|^2 + \nabla_\nu|\chi_j|^2 / |\chi_j|^2
        self.K = np.zeros((self.ntrajs, self.nst, self.nst)) # Decoherence term from total quantum momentum: \sum_\nu 1/(2M_\nu) \mathcal{P}_\nu \cdot \nabla_\nu (S_i - S_j)
        self.K_bo = np.zeros((self.ntrajs, self.nst, self.nst)) # Decoherence term from projected quantum momentum: \sum_\nu 1/(2M_\nu) \mathcal{G}_{\nu,ij} \cdot \nabla_\nu (S_i - S_j)
        self.mom = np.zeros((self.ntrajs, self.nst, self.nat_qm, self.ndim)) # state-wise momentum
        self.dS = np.zeros((self.ntrajs, self.nst, self.nat_qm, self.ndim)) # phase term used in phase correction
        self.d2e = np.zeros((self.ntrajs, self.nst, self.nat_qm)) # Laplacian of BO energy
        self.d2S = np.zeros((self.ntrajs, self.nst, self.nat_qm)) # Laplacian of S_i's 
        self.alpha = np.zeros((self.nst_pair, self.nat_qm)) # Scaling factor
        self.beta = np.zeros((self.nst_pair, self.nat_qm)) # Shift factor
        self.etot0 = np.zeros((self.ntrajs)) # Total energy at t=0
        self.l_coh = np.zeros((self.ntrajs, self.nst), dtype=bool) # Flag that a trajectory is in coherence
        self.l_first = np.zeros((self.ntrajs, self.nst), dtype=bool) # Flag that a trajectory become coherent

        # Initialize variables to calculate quantum momentum 
        self.count_ntrajs = np.ones((self.ntrajs, self.nat_qm, self.ndim)) * self.ntrajs
        self.sigma = np.zeros((self.nst, self.nat_qm, self.ndim)) 
        self.slope = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        self.intercept = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        self.slope_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim))
        self.intercept_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim))
        self.g_I = np.zeros((self.ntrajs)) # |\chi|^2
        self.g_i_I = np.ones((self.nst, self.ntrajs)) # |\chi_i|^2
        self.g_i_IJ = np.ones((self.nst, self.ntrajs, self.ntrajs)) # Gaussian basis for |\chi_i|^2 when l_traj_gaussian=True
        self.center = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        self.center_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim))
        self.avg_R = np.zeros((self.nst, self.nat_qm, self.ndim))
        self.pseudo_pop = np.zeros((self.nst, self.ntrajs))

        # Determine parameters to calculate decoherenece effect
        self.small = 1.0E-08

        self.rho_threshold = rho_threshold
        self.upper_th = 1. - self.rho_threshold
        self.lower_th = self.rho_threshold

        self.artifact_expon = artifact_expon

        self.l_en_cons = l_en_cons
        self.dotpopnac = np.zeros((self.ntrajs, self.nst))
        self.dotpopdec = np.zeros((self.ntrajs, self.nst))
        
        self.l_crunch = l_crunch
        self.l_real_pop = l_real_pop
        self.l_dc_w_mom = l_dc_w_mom
        self.l_traj_gaussian = l_traj_gaussian
        self.l_lap = l_lap
        self.t_pc = t_pc
        self.t_cons = t_cons
        self.l_etot0 = l_etot0

        # Variables for aborting dynamics when all trajectories reach asymptotic region
        self.l_asymp = l_asymp
        self.x_fin = x_fin
        
        # Initialize event to print
        self.event = {"DECO": []}

    def run(self, qm, mm=None, output_dir="./", l_save_qm_log=False, l_save_mm_log=False, l_save_scr=True, restart=None):
        """ Run MQC dynamics according to CTMQC dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string output_dir: Name of directory where outputs to be saved.
            :param boolean l_save_qm_log: Logical for saving QM calculation log
            :param boolean l_save_mm_log: Logical for saving MM calculation log
            :param boolean l_save_scr: Logical for saving scratch directory
            :param string restart: Option for controlling dynamics restarting
        """
        # Initialize PyUNIxMD
        abs_path_output_dir = os.path.join(os.getcwd(), output_dir)
        base_dirs, unixmd_dirs, qm_log_dirs, mm_log_dirs =\
             self.run_init(qm, mm, output_dir, l_save_qm_log, l_save_mm_log, l_save_scr, restart)

        bo_list = [ist for ist in range(self.nst)]
        qm.calc_coupling = True

        self.print_init(qm, mm, restart)

        if (restart == None):
            # Calculate initial input geometry for all trajectories at t = 0.0 s
            self.istep = -1
            for itraj in range(self.ntrajs):
                self.mol = self.mols[itraj]

                self.mol.reset_bo(qm.calc_coupling)
                qm.get_data(self.mol, base_dirs[itraj], bo_list, self.dt, self.istep, calc_force_only=False, md=self, itraj=itraj)

                # TODO: QM/MM
                self.mol.get_nacme()
                
                self.check_decoherence(itraj)
                self.check_coherence(itraj)
                
                self.update_energy()
                
                self.get_state_mom(itraj)
                
                self.get_phase(itraj)

            if (self.t_pc != 0):
                self.get_dS()

            self.calculate_qmom()
            
            if (self.l_lap):
                self.get_d2S()
            
            self.set_avg_pop_cons()

            for itraj in range(self.ntrajs):

                self.mol = self.mols[itraj]

                self.write_md_output(itraj, unixmd_dirs[itraj], self.istep)

                self.print_step(self.istep, itraj)

        #TODO: restart
        elif (restart == "write"):
            # Reset initial time step to t = 0.0 s
            self.istep = -1
            for itraj in range(self.ntrajs):
                self.write_md_output(itraj, unixmd_dirs[itraj], self.istep)
                self.print_step(self.istep, itraj)

        elif (restart == "append"):
            # Set initial time step to last successful step of previous dynamics
            self.istep = self.fstep

        self.istep += 1

        # Main MD loop
        for istep in range(self.istep, self.nsteps):
            for itraj in range(self.ntrajs):
                self.mol = self.mols[itraj]

                self.calculate_force(itraj)
                self.cl_update_position()

                self.mol.backup_bo()
                self.mol.reset_bo(qm.calc_coupling)

                qm.get_data(self.mol, base_dirs[itraj], bo_list, self.dt, self.istep, calc_force_only=False, md=self, itraj=itraj)

                if (not self.mol.l_nacme and self.l_adj_nac):
                    self.mol.adjust_nac()

                #TODO: QM/MM

                self.calculate_force(itraj)
                self.cl_update_velocity()

                self.mol.get_nacme()
                
                self.update_energy()

                el_run(self, itraj)

                #TODO: thermostat
                #if (self.thermo != None):
                #    self.thermo.run(self)

                self.check_decoherence(itraj)
                self.check_coherence(itraj)
                
                self.update_energy()

                self.get_state_mom(itraj)
                
                self.get_phase(itraj)
            
            if (self.t_pc != 0):
                self.get_dS()
            
            self.calculate_qmom()
            
            if (self.l_lap):
                self.get_d2S()
            
            self.set_avg_pop_cons()
            
            for itraj in range(self.ntrajs):
                self.mol = self.mols[itraj]

                if ((istep + 1) % self.out_freq == 0):
                    self.write_md_output(itraj, unixmd_dirs[itraj], istep)
                    self.print_step(istep, itraj)
                if (istep == self.nsteps - 1):
                    self.write_final_xyz(unixmd_dirs[itraj], istep)

            self.fstep = istep
            #restart_file = os.path.join(abs_path_output_dir, "RESTART.bin")
            #with open(restart_file, 'wb') as f:
            #    pickle.dump({'qm':qm, 'md':self}, f)

            for itraj in range(self.ntrajs):
                l_abort = True
                det = self.mols[itraj].pos[0, 0] * self.mols[itraj].vel[0, 0]
                if (self.l_asymp and det > 0. and np.abs(self.mols[itraj].pos[0, 0]) > np.abs(self.x_fin)):
                    pass
                else:
                    l_abort = False
                    break
            
            if(l_abort):
                break 

        # Delete scratch directory
        if (not l_save_scr):
            for itraj in range(self.ntrajs):
                tmp_dir = os.path.join(unixmd_dirs[itraj], "scr_qm")
                if (os.path.exists(tmp_dir)):
                    shutil.rmtree(tmp_dir)
    
    def get_dS(self):
        self.dS[:, :, :, :] = self.mom[:, :, :, :]
    
    def get_d2S(self):
        
        # Calculate d2S by integrating Laplacian of BO energy 
        for itraj in range(self.ntrajs):
            for ist in range(self.nst):
                #if (self.l_coh[itraj, ist]):
                #    self.d2S[itraj, ist, :] -= self.d2e[itraj, ist, :] * self.dt
                self.d2S[itraj, ist, :] -= self.d2e[itraj, ist, :] * self.dt
            
    def set_avg_pop_cons(self):
        
        # Rescale d2S
        if (self.t_cons == 1 and self.l_lap):
            deno = np.zeros((self.nst_pair, self.nat_qm))
            numer = np.zeros((self.nst_pair, self.nat_qm))
            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist+1, self.nst):
                    index_lk += 1
                    for iat in range(self.nat_qm):
                        deno[index_lk, iat] = 0.0
                        numer[index_lk, iat] = 0.0
                        for itraj in range(self.ntrajs):
                            if (self.l_coh[itraj, ist] and self.l_coh[itraj, jst]):
                                if (self.l_crunch):
                                    numer[index_lk, iat] += \
                                        - np.sum((self.qmom_bo[itraj, index_lk, iat, :] - self.qmom[itraj, iat, :]) * (self.phase[itraj, ist, iat, :] - self.phase[itraj, jst, iat, :])) \
                                        * self.mols[itraj].rho.real[ist, ist] * self.mols[itraj].rho.real[jst, jst]
                                else:
                                    numer[index_lk, iat] += \
                                        - np.sum((self.qmom[itraj, iat, :]) * (self.phase[itraj, ist, iat, :] - self.phase[itraj, jst, iat, :])) \
                                        * self.mols[itraj].rho.real[ist, ist] * self.mols[itraj].rho.real[jst, jst]
                                deno[index_lk, iat] += (self.d2S[itraj, ist, iat] - self.d2S[itraj, jst, iat]) * self.mols[itraj].rho.real[ist, ist] * self.mols[itraj].rho.real[jst, jst]

                        if (abs(deno[index_lk, iat]) < self.small):
                            self.alpha[index_lk, iat] = 0.0
                        else:
                            self.alpha[index_lk, iat] = numer[index_lk, iat] / deno[index_lk, iat]

        # Or add const: beta (\Delta_{ij} in the paper and note)
        elif (self.t_cons == 2):
            self.alpha[:, :] = 1.0
            deno = np.zeros((self.nst_pair))
            numer = np.zeros((self.nst_pair, self.nat_qm))
            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist+1, self.nst):
                    index_lk += 1
                    deno[index_lk] = 0.0
                    for itraj in range(self.ntrajs):
                        deno[index_lk] +=  self.mols[itraj].rho.real[ist, ist] * self.mols[itraj].rho.real[jst, jst]
                    
                    if (abs(deno[index_lk]) < self.small):
                        self.beta[index_lk, :] = 0.0
                    else:
                        for iat in range(self.nat_qm):
                            numer[index_lk, iat] = 0.0
                            for itraj in range(self.ntrajs):
                                if (self.l_coh[itraj, ist] and self.l_coh[itraj, jst]):
                                    if (self.l_crunch):
                                        numer[index_lk, iat] += \
                                            - np.sum((self.qmom_bo[itraj, index_lk, iat, :] - self.qmom[itraj, iat, :]) * (self.phase[itraj, ist, iat, :] - self.phase[itraj, jst, iat, :])) \
                                            * self.mols[itraj].rho.real[ist, ist] * self.mols[itraj].rho.real[jst, jst]
                                    else:
                                        numer[index_lk, iat] += \
                                            - np.sum((self.qmom[itraj, iat, :]) * (self.phase[itraj, ist, iat, :] - self.phase[itraj, jst, iat, :])) \
                                            * self.mols[itraj].rho.real[ist, ist] * self.mols[itraj].rho.real[jst, jst]
                                    if (self.l_lap):
                                        numer[index_lk, iat] += - (self.d2S[itraj, ist, iat] - self.d2S[itraj, jst, iat]) * self.mols[itraj].rho.real[ist, ist] * self.mols[itraj].rho.real[jst, jst]

                            self.beta[index_lk, iat] = numer[index_lk, iat] / deno[index_lk]
        else:
            self.alpha[:, :] = 1.0
            self.beta[:, :] = 0.0

    def get_state_mom(self, itrajectory):
        
        if (self.istep == -1):
            self.etot0[itrajectory] = self.mol.etot 

        for ist in range(self.nst):
            if (self.l_etot0):
                alpha = (self.etot0[itrajectory] - self.mol.states[ist].energy) / self.mol.ekin
            else:
                alpha = (self.mol.etot - self.mol.states[ist].energy) / self.mol.ekin

            if (alpha < 0):
                alpha = 0.
            for iat in range(self.nat_qm):
                self.mom[itrajectory, ist, iat, :] = np.sqrt(alpha) * self.mol.vel[iat, :] * self.mol.mass[iat]

    def calculate_force(self, itrajectory):
        """ Routine to calculate force

            :param integer itrajectory: Index for trajectories
        """
        self.rforce = np.zeros((self.nat_qm, self.ndim))

        # Derivatives of energy
        for ist, istate in enumerate(self.mols[itrajectory].states):
            self.rforce += istate.force * self.mol.rho.real[ist, ist]

        # Non-adiabatic forces 
        for ist in range(self.nst):
            for jst in range(ist + 1, self.nst):
                self.rforce += 2. * self.mol.nac[ist, jst] * self.mol.rho.real[ist, jst] \
                    * (self.mol.states[ist].energy - self.mol.states[jst].energy)

        # CT forces = -\sum_{i,j} |C_iC_j|^2 K_lk_{i, j} * (f_i - f_j)
        ctforce = np.zeros((self.nat_qm, self.ndim))
        if (self.l_crunch):
            for ist in range(self.nst):
                for jst in range(self.nst):
                    ctforce -= (self.K_bo[itrajectory, ist, jst] - self.K[itrajectory, ist, jst])* \
                        (self.phase[itrajectory, ist] - self.phase[itrajectory, jst]) * \
                        self.mol.rho.real[ist, ist] * self.mol.rho.real[jst, jst]
            if (self.l_lap): # only one-dimensional, same atom term is considered.
                index_lk = -1
                for ist in range(self.nst):
                    for jst in range(ist+1, self.nst):
                        index_lk += 1
                        ctforce -= 2.0 * np.sum(1. / self.mol.mass[0:self.nat_qm] * (self.d2S[itrajectory, ist, :] - self.d2S[itrajectory, jst, :]) * self.alpha[index_lk, :]) * \
                            (self.phase[itrajectory, ist] - self.phase[itrajectory, jst]) * \
                            self.mol.rho.real[ist, ist] * self.mol.rho.real[jst, jst]
            if (self.t_cons == 2):
                index_lk = -1
                for ist in range(self.nst):
                    for jst in range(ist+1, self.nst):
                        index_lk += 1
                        ctforce -= 2.0 * np.sum(1. / self.mol.mass[0:self.nat_qm] * (self.beta[index_lk, :])) * \
                            (self.phase[itrajectory, ist] - self.phase[itrajectory, jst]) * \
                            self.mol.rho.real[ist, ist] * self.mol.rho.real[jst, jst]

        else:
            for ist in range(self.nst):
                for jst in range(self.nst):
                    ctforce -= self.K[itrajectory, ist, jst] * \
                        (self.phase[itrajectory, ist] - self.phase[itrajectory, jst]) * \
                        self.mol.rho.real[ist, ist] * self.mol.rho.real[jst, jst]

        # Finally, force is Ehrenfest force + CT force
        self.rforce += ctforce

    def update_energy(self):
        """ Routine to update the energy of molecules in CTMQC dynamics
        """
        # Update kinetic energy
        self.mol.update_kinetic()
        self.mol.epot = 0.
        for ist, istate in enumerate(self.mol.states):
            self.mol.epot += self.mol.rho.real[ist, ist] * istate.energy
        
        if (self.l_en_cons and not (self.istep == -1)):
            alpha = (self.mol.etot - self.mol.epot)
            factor = alpha / self.mol.ekin

            self.mol.vel *= np.sqrt(factor)
            self.mol.update_kinetic()

        self.mol.etot = self.mol.epot + self.mol.ekin

    def get_phase(self, itrajectory):
        """ Routine to calculate phase

            :param integer itrajectory: Index for trajectories
        """
        if (self.l_dc_w_mom):
            for ist in range(self.nst):
                self.phase[itrajectory, ist, :, :] = self.mom[itrajectory, ist, :, :]
        else:
            for ist in range(self.nst):
                self.phase[itrajectory, ist] += self.mol.states[ist].force * self.dt
                #rho = self.mol.rho[ist, ist].real
                #if (rho > self.upper_th or rho < self.lower_th):
                #    self.phase[itrajectory, ist] = np.zeros((self.nat_qm, self.ndim))
                #else:
                #    self.phase[itrajectory, ist] += self.mol.states[ist].force * self.dt

    def check_coherence(self, itrajectory):
        """ Routine to check coherence among BO states

            :param integer itrajectory: Index for trajectories
        """
        count = 0
        tmp_st = ""
        rho_tmp = np.zeros((self.ntrajs, self.nst))
        
        for itraj in range(self.ntrajs):
            for ist in range(self.nst):
                rho_tmp[itraj, ist] = self.mols[itraj].rho.real[ist, ist]
        
        for ist in range(self.mol.nst):
            avg_rho = np.sum(rho_tmp[:, ist]) / self.ntrajs
            rho = rho_tmp[itrajectory, ist]
            
            if ((rho > self.upper_th or rho < self.lower_th) or \
                (avg_rho > self.upper_th or avg_rho < self.lower_th)):
                self.l_coh[itrajectory, ist] = False
            else:
                if (self.l_coh[itrajectory, ist]):
                    self.l_first[itrajectory, ist] = False
                else:
                    self.l_first[itrajectory, ist] = True
                    tmp_st += f"{ist}, "
                self.l_coh[itrajectory, ist] = True
                count += 1
        
        if (count < 2):
            self.l_coh[itrajectory, :] = False
            self.l_first[itrajectory, :] = False
            tmp_st = ""
        
        if (len(tmp_st) >= 1):
            tmp_st = tmp_st.rstrip(', ')
            self.event["DECO"].append(f"{itrajectory + 1:8d}: {tmp_st} states are in coherence")

    def check_decoherence(self, itrajectory):
        """ Routine to check decoherence among BO states

            :param integer itrajectory: Index for trajectories
        """
        for ist in range(self.mol.nst):
            if (self.l_coh[itrajectory, ist]):
                rho = self.mol.rho.real[ist, ist]
                if (rho > self.upper_th):
                    #self.set_decoherence(ist)
                    self.event["DECO"].append(f"{itrajectory + 1:8d}: decohered to {ist} state")
                    return
    
    def set_decoherence(self, one_st):
        """ Routine to reset coefficient/density if the state is decohered

            :param integer one_st: State index that its population is one
        """
        self.mol.rho = np.zeros((self.mol.nst, self.mol.nst), dtype=np.complex64)
        self.mol.rho[one_st, one_st] = 1. + 0.j
        
        if (self.elec_object == "coefficient"):
            for ist in range(self.mol.nst):
                if (ist == one_st):
                    self.mol.states[ist].coef /= np.absolute(self.mol.states[ist].coef).real
                else:
                    self.mol.states[ist].coef = 0. + 0.j

    def calculate_qmom(self):
        """ Routine to calculate quantum momentum
        """
        # _lk means state_pair dependency.
        # i and j are trajectory index.
        # -------------------------------------------------------------------
        # 1. Calculate variances for each trajectory
        self.calculate_sigma()

        # 2. Calculate slope
        self.calculate_slope()

        # 3. Calculate the center of quantum momentum
        self.calculate_center()
        
        # 4. Compute quantum momentum
        # G_{\nu, ij} = (\nabla_\nu|\chi_i|^2 / |\chi_i|^2  + \nabla_\nu|\chi_j|^2/|\chi_j|^2)
        # and/or
        # P_{\nu} = \nabla_\nu|\chi|^2 / |\chi|^2
        for itraj in range(self.ntrajs):
            self.qmom[itraj] = self.slope[itraj] * self.mols[itraj].pos - self.intercept[itraj]
            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    index_lk += 1
                    if (self.l_crunch):
                        self.qmom_bo[itraj, index_lk] = self.slope_bo[itraj, index_lk] * self.mols[itraj].pos - self.intercept_bo[itraj, index_lk]

        # 5. Calculate
        # K_bo = 0.5 * G_{\nu, ij}/M \cdot D_{ij} 
        # and/or
        # K = 0.5 * P_{\nu}/M \cdot D_{ij}
        self.K = np.zeros((self.ntrajs, self.nst, self.nst))
        self.K_bo = np.zeros((self.ntrajs, self.nst, self.nst))
        for itraj in range(self.ntrajs):
            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    index_lk += 1
                    if (self.l_coh[itraj, ist] and self.l_coh[itraj, jst]):
                        self.K[itraj, ist, jst] = 0.5 * np.sum(1. / self.mol.mass[0:self.nat_qm] * \
                            np.sum(self.qmom[itraj] * (self.phase[itraj, ist] - self.phase[itraj, jst]), axis = 1))
                        self.K[itraj, jst, ist] = - self.K[itraj, ist, jst]
                        if (self.l_crunch):
                            self.K_bo[itraj, ist, jst] = 0.5 * np.sum(1. / self.mol.mass[0:self.nat_qm] * \
                                np.sum(self.qmom_bo[itraj, index_lk] * (self.phase[itraj, ist] - self.phase[itraj, jst]), axis = 1))
                            
                            #if (self.l_lap):
                            #    self.K_bo[itraj, ist, jst] += 0.5 * np.sum(1. / self.mol.mass[0:self.nat_qm] * (self.d2S[itraj, ist, :] - self.d2S[itraj, jst, :]))
                            
                            self.K_bo[itraj, jst, ist] = - self.K_bo[itraj, ist, jst]


    def calculate_sigma(self):
        """ Routine to calculate variances for each trajectories
        """
        pos = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        rho = np.zeros((self.ntrajs, self.nst))
        for itraj in range(self.ntrajs):
            pos[itraj, :, :] = self.mols[itraj].pos[:, :]
            for ist in range(self.nst):
                rho[itraj, ist] = self.mols[itraj].rho.real[ist, ist]

        for ist in range(self.nst):
            rho_tmp = np.sum(rho[:, ist])
            
            if (rho_tmp / self.ntrajs < self.lower_th):
                self.avg_R[ist, :, :] = 0.0
                self.sigma[ist, :, :] = np.inf
            else: 
                self.avg_R[ist, :, :] = np.tensordot(pos[:, :, :], rho[:, ist], axes=(0,0)) / rho_tmp
                self.sigma[ist, :, :] = np.sqrt(np.tensordot(pos[:, :, :] ** 2, rho[:, ist], axes=(0,0)) / rho_tmp - self.avg_R[ist, :, :] ** 2)
        
        if (self.l_traj_gaussian):
            
            #iqr = np.zeros((self.nst, self.nat_qm, self.ndim))
            #rho_sort = np.zeros((self.ntrajs, self.nst))
            #pos_sort = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
            for ist in range(self.nst):
                rho_tmp = np.sum(rho[:, ist]) / self.ntrajs
                if (rho_tmp < self.lower_th):
                    self.avg_R[ist, :, :] = 0.0
                    self.sigma[ist, :, :] = np.inf
                else:
                    #for iat in range(self.nat_qm):
                    #    for idim in range(self.ndim):
                    #        idx = np.argsort(pos[:, iat, idim])
                    #        pos_sort[:, iat, idim] = pos[idx, iat, idim]
                    #        rho_sort[:, ist] = rho[idx, ist]
                    #        weighted_cdf = np.cumsum(rho[:, ist])
                    #        weighted_cdf /= weighted_cdf[-1]
                    #        q25, q75 = np.interp([0.25, 0.75], weighted_cdf, pos_sort[:, iat, idim])
                    #        iqr[ist, iat, idim] = q75 - q25

                    #self.sigma[ist, :, :] = np.minimum(iqr[ist, :, :] / 1.34, self.sigma[ist, :, :])
                    #self.sigma[ist, :, :] *= 0.9 * (rho_tmp * self.ntrajs) ** (-self.artifact_expon) # N --> N of ist.
                    self.sigma[ist, :, :] *= 1.06 * (rho_tmp * self.ntrajs) ** (-self.artifact_expon) # N --> N of ist.
                #self.sigma[ist, :, :] *= (self.ntrajs) ** (-self.artifact_expon)



    def calculate_slope(self):
        """ Routine to calculate slope
        """
        
        pos = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        rho = np.zeros((self.ntrajs, self.nst))
        for itraj in range(self.ntrajs):
            pos[itraj, :, :] = self.mols[itraj].pos[:, :]
            for ist in range(self.nst):
                rho[itraj, ist] = self.mols[itraj].rho.real[ist, ist]
        
        self.g_I = np.zeros((self.ntrajs)) 
        self.g_i_I = np.ones((self.nst, self.ntrajs))
        self.g_i_IJ = np.ones((self.nst, self.ntrajs, self.ntrajs))

        for ist in range(self.nst):
            rho_tmp = np.sum(rho[:, ist]) / self.ntrajs
            if (rho_tmp < self.lower_th):
                self.g_i_I[ist, :] = 0.0
            
            else: 
                if (self.l_traj_gaussian):
                    for itraj in range(self.ntrajs):
                        self.g_i_I[ist, itraj] = 0.0
                        for jtraj in range(self.ntrajs):
                            for iat in range(self.nat_qm):
                                for idim in range(self.ndim):
                                    self.g_i_IJ[ist, itraj, jtraj] *= gaussian1d(self.mols[itraj].pos[iat, idim], 1., \
                                        self.sigma[ist, iat, idim], self.mols[jtraj].pos[iat, idim])
                            
                            self.g_i_IJ[ist, itraj, jtraj] *= self.mols[jtraj].rho.real[ist, ist] / self.ntrajs

                            self.g_i_I[ist, itraj] += self.g_i_IJ[ist, itraj, jtraj] # |\chi_i|^2
                    
                else:
                    for itraj in range(self.ntrajs):
                        for iat in range(self.nat_qm):
                            for idim in range(self.ndim):
                                # gaussian1d(x, pre-factor, sigma, mean)
                                # gaussian1d(R^{itraj}, 1.0, sigma^{jtraj}, R^{jtraj})
                                self.g_i_I[ist, itraj] *= gaussian1d(self.mols[itraj].pos[iat, idim], 1., \
                                    self.sigma[ist, iat, idim], self.avg_R[ist, iat, idim])
                
                    self.g_i_I[ist, :] *= rho_tmp # |\chi_i|^2

        self.g_I[:] = np.sum(self.g_i_I[:, :], axis=0) # |\chi|^2
        
        self.pseudo_pop = np.zeros((self.nst, self.ntrajs))
        
        if (self.l_real_pop):
            for itraj in range(self.ntrajs):
                self.pseudo_pop[:, itraj] = rho[itraj, :]
        else:
            for itraj in range(self.ntrajs):
                if (self.g_I[itraj] < self.small):
                    pass
                else:
                    for ist in range(self.nst):
                        self.pseudo_pop[ist, itraj] = self.g_i_I[ist, itraj] / self.g_I[itraj]

        for itraj in range(self.ntrajs):
            for iat in range(self.nat_qm):
                for isp in range(self.ndim):
                    self.slope[itraj, iat, isp] = - 1.0 * np.sum(self.pseudo_pop[:, itraj]  / self.sigma[:, iat, isp] ** 2) # slope for total quantum momentum
        
        if (self.l_crunch):
            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist+1, self.nst):
                    index_lk += 1
                    for iat in range(self.nat_qm):
                        for isp in range(self.ndim):
                            if (self.sigma[ist, iat, isp] ** 2 < self.small or self.sigma[jst, iat, isp] ** 2 < self.small):
                                self.slope_bo[:, index_lk, iat, isp] = 0.0
                            else:
                                for itraj in range(self.ntrajs):
                                    self.slope_bo[itraj, index_lk, iat, isp] = - 1.0 * \
                                        (1.0 / self.sigma[ist, iat, isp] ** 2 + 1.0 / self.sigma[jst, iat, isp] ** 2) # slope for projected quantum momentum
        
    def calculate_center(self):
        """ Routine to calculate center or intercept of quantum momentum
        """
        rho = np.zeros((self.ntrajs, self.nst))
        for itraj in range(self.ntrajs):
            for ist in range(self.nst):
                rho[itraj, ist] = self.mols[itraj].rho[ist, ist].real

        # Center is not actually used. Just to check.
        # Intercepts for total and projected quantum momenta
        for itraj in range(self.ntrajs):
            for iat in range(self.nat_qm):
                for isp in range(self.ndim):
                    if (self.l_traj_gaussian):
                        self.intercept[itraj, iat, isp] = 0.0
                        if (self.g_I[itraj] / self.ntrajs < self.small):
                            self.intercept[itraj, iat, isp] = 0.0
                            self.center[itraj, iat, isp] = self.mols[itraj].pos[iat, isp]
                        else:
                            for jtraj in range(self.ntrajs):
                                self.intercept[itraj, iat, isp] += -1.0 * \
                                    np.sum((self.g_i_IJ[:, itraj, jtraj] * self.mols[jtraj].pos[iat, isp]) / self.sigma[:, iat, isp] ** 2) / self.g_I[itraj]
                    else:
                        self.intercept[itraj, iat, isp] = - 1.0 * \
                            np.sum(self.pseudo_pop[:, itraj] * (self.avg_R[:, iat, isp] / self.sigma[:, iat, isp] ** 2))
            
                    if (abs(self.slope[itraj, iat, isp]) < self.small):
                        self.center[itraj, iat, isp] = self.mols[itraj].pos[iat, isp]
                    else:
                        self.center[itraj, iat, isp] = self.intercept[itraj, iat, isp] / self.slope[itraj, iat, isp]

        if (self.l_crunch):
            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist+1, self.nst):
                    index_lk += 1
                    for itraj in range(self.ntrajs):
                        for iat in range(self.nat_qm):
                            for isp in range(self.ndim):
                                if (self.sigma[ist, iat, isp] ** 2 < self.small or self.sigma[jst, iat, isp] ** 2 < self.small):
                                    self.intercept_bo[itraj, index_lk, :, :] = 0.0
                                    self.center_bo[itraj, index_lk, :, :] = self.mols[itraj].pos[:, :]
                                else:
                                    if (self.l_traj_gaussian):
                                        self.intercept_bo[itraj, index_lk, iat, isp] = 0.0
                                        if (self.g_i_I[ist, itraj] / self.ntrajs < self.small or self.g_i_I[jst, itraj] / self.ntrajs < self.small):
                                            self.intercept_bo[itraj, index_lk, iat, isp] = 0.0
                                            self.slope_bo[itraj, index_lk, iat, isp] = 0.0
                                        else:
                                            for jtraj in range(self.ntrajs):
                                                self.intercept_bo[itraj, index_lk, iat, isp] += -1.0 *\
                                                    (self.g_i_IJ[ist, itraj, jtraj] * self.mols[jtraj].pos[iat, isp]) / self.sigma[ist, iat, isp] ** 2 / self.g_i_I[ist, itraj]
                                                self.intercept_bo[itraj, index_lk, iat, isp] += -1.0 *\
                                                    (self.g_i_IJ[jst, itraj, jtraj] * self.mols[jtraj].pos[iat, isp]) / self.sigma[jst, iat, isp] ** 2 / self.g_i_I[jst, itraj]
                                        
                                    else:
                                        self.intercept_bo[itraj, index_lk, iat, isp] = - 1.0 * \
                                            (self.avg_R[ist, iat, isp] / self.sigma[ist, iat, isp] ** 2 + self.avg_R[jst, iat, isp] / self.sigma[jst, iat, isp] ** 2)                                
                                if (abs(self.slope_bo[itraj, index_lk, iat, isp]) < self.small):
                                    self.center_bo[itraj, index_lk, iat, isp] = self.mols[itraj].pos[iat, isp]
                                else:
                                    self.center_bo[itraj, index_lk, iat, isp] = self.intercept_bo[itraj, index_lk, iat, isp] / self.slope_bo[itraj, index_lk, iat, isp]

    def check_istates(self):
        """ Routine to check istates and init_coefs
        """
        if (self.istates != None):
            if (isinstance(self.istates, list)):
                if (len(self.istates) != self.ntrajs):
                    error_message = "Number of elements of initial states must be equal to number of trajectories!"
                    error_vars = f"len(istates) = {len(self.istates)}, ntrajs = {self.ntrajs}"
                    raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
                else:
                    self.init_coefs = [None] * self.ntrajs
            else:
                error_message = "The type of initial states must be list!"
                error_vars = f"istates = {self.istates}"
                raise TypeError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
        else:
            if (self.init_coefs == None):
                error_message = "Either initial states or coefficients must be given!"
                error_vars = f"istates = {self.istates}, init_coefs = {self.init_coefs}"
                raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
            else:
                if (isinstance(self.init_coefs, list)):
                    if (len(self.init_coefs) != self.ntrajs):
                        error_message = "Number of elements of initial coefficients must be equal to number of trajectories!"
                        error_vars = f"len(init_coefs) = {len(self.init_coefs)}, ntrajs = {self.ntrajs}"
                        raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
                    else:
                        self.istates = [None] * self.ntrajs
                else:
                    error_message = "Type of initial coefficients must be list!"
                    error_vars = f"init_coefs = {self.init_coefs}"
                    raise TypeError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

    def write_md_output(self, itrajectory, unixmd_dir, istep):
        """ Write output files

            :param integer itrajectory: Index for trajectories
            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step
        """
        # Write the common part
        super().write_md_output(unixmd_dir, istep)

        # Write time-derivative BO population
        self.write_dotpop(itrajectory, unixmd_dir, istep)

        # Write decoherence information
        self.write_dec(itrajectory, unixmd_dir, istep)

    def write_dotpop(self, itrajectory, unixmd_dir, istep):
        """ Write time-derivative BO population

            :param integer itrajectory: Index for trajectories
            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step
        """
        if (self.verbosity >= 1):
            # Write NAC term in DOTPOPNAC
            tmp = f'{istep + 1:9d}' + "".join([f'{pop:15.8f}' for pop in self.dotpopnac[itrajectory]])
            typewriter(tmp, unixmd_dir, "DOTPOPNAC", "a")

            # Write decoherence term in DOTPOPDEC
            tmp = f'{istep + 1:9d}' + "".join([f'{pop:15.8f}' for pop in self.dotpopdec[itrajectory]])
            typewriter(tmp, unixmd_dir, "DOTPOPDEC", "a")

    def write_dec(self, itrajectory, unixmd_dir, istep):
        """ Write CT-based decoherence information

            :param integer itrajectory: Index for trajectories
            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step
        """
        if (self.verbosity >= 1):
            # Write K_lk
            for ist in range(self.nst):
                for jst in range(self.nst):
                    if (ist != jst):
                        tmp = f'{istep + 1:9d}{self.K[itrajectory, ist, jst]:15.8f}'
                        typewriter(tmp, unixmd_dir, f"K_{ist}_{jst}", "a")
                        
                        tmp = f'{istep + 1:9d}{self.K_bo[itrajectory, ist, jst]:15.8f}'
                        typewriter(tmp, unixmd_dir, f"K_BO_{ist}_{jst}", "a")

        # Write detailed quantities related to decoherence
        if (self.verbosity >= 2):
            tmp = f'{istep + 1:9d}' + "".join([f'{pop:15.8f}' for pop in self.pseudo_pop[:, itrajectory]])
            typewriter(tmp, unixmd_dir, "PSEUDOPOP", "a")
            if (itrajectory == 0):
                for ist in range(self.nst):
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}sigma_x{"":5s}sigma_y{"":5s}sigma_z{"":5s}' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.sigma[ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"SIGMA_{ist}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s} Center (au)' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.avg_R[ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"CENTER_{ist}", "a")
                    
                    #tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}sigma_x{"":5s}sigma_y{"":5s}sigma_z{"":5s}count_ntrajs' + \
                    #    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    #    "".join([f'{self.sigma[itrajectory, ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) + \
                    #    "".join([f'{self.count_ntrajs[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    #typewriter(tmp, unixmd_dir, f"SIGMA", "a")
            
            # Write quantum momenta
            tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum (au)' + \
                "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                "".join([f'{self.qmom[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
            typewriter(tmp, unixmd_dir, f"QMOM", "a")
            tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Slope' + \
                "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                "".join([f'{self.slope[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
            typewriter(tmp, unixmd_dir, f"SLOPE", "a")
            
            tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Intercept' + \
                "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                "".join([f'{self.intercept[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
            typewriter(tmp, unixmd_dir, f"INTERCEPT", "a")

            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    index_lk += 1
                    #tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum center (au)' + \
                    #    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    #    "".join([f'{self.center_lk[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    #typewriter(tmp, unixmd_dir, f"CENTER_{ist}_{jst}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum (au)' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.qmom_bo[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"QMOM_BO_{ist}_{jst}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Slope' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.slope_bo[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"SLOPE_BO_{ist}_{jst}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Intercept' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.intercept_bo[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"INTERCEPT_BO_{ist}_{jst}", "a")

            # Write Phase
            for ist in range(self.mol.nst):
                tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Phase (au)' + \
                    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    "".join([f'{self.phase[itrajectory, ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                typewriter(tmp, unixmd_dir, f"PHASE_{ist}", "a")
            
            # Write state momentum
            for ist in range(self.mol.nst):
                tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum (au)' + \
                    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    "".join([f'{self.mom[itrajectory, ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                typewriter(tmp, unixmd_dir, f"MOM_{ist}", "a")

    def print_init(self, qm, mm, restart):
        """ Routine to print the initial information of dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string restart: Option for controlling dynamics restarting
        """
        # Print initial information about molecule, qm, mm and thermostat
        super().print_init(qm, mm, restart)

        # Print CTMQC info.
        ct_info = textwrap.dedent(f"""\
        {"-" * 68}
        {"CTMQC Information":>43s}
        {"-" * 68}
          rho_threshold            = {self.rho_threshold:>16f}
          l_crunch                 = {self.l_crunch:>16}
          l_dc_w_mom               = {self.l_dc_w_mom:>16}
          l_traj_gaussian          = {self.l_traj_gaussian:>16}
          t_cons                   = {self.t_cons:>16d}
          l_etot0                  = {self.l_etot0:>16}
          l_lap                    = {self.l_lap:>16}
        """)

        print (ct_info, flush=True)

        # Print istate
        istate_info = textwrap.dedent(f"""\
        {"-" * 68}
        {"Initial State Information":>43s}
        {"-" * 68}
        """)
        istate_info += f"  istates (1:{self.ntrajs})             =\n"
        nlines = self.ntrajs // 6
        if (self.ntrajs % 6 == 0):
            nlines -= 1

        for iline in range(nlines + 1):
            iline1 = iline * 6
            iline2 = (iline + 1) * 6
            if (iline2 > self.ntrajs):
                iline2 = self.ntrajs
            istate_info += f"  {iline1 + 1:>4d}:{iline2:<4d};"
            istate_info += "".join([f'{str(istate):7s}' for istate in self.istates[iline1:iline2]])
            istate_info += "\n"
        print (istate_info, flush=True)

        # Print dynamics information for start line
        dynamics_step_info = textwrap.dedent(f"""\

        {"-" * 118}
        {"Start Dynamics":>65s}
        {"-" * 118}
        """)

        # Print INIT for each trajectory at each step
        INIT = f" #INFO_TRAJ{'STEP':>8s}{'Kinetic(H)':>15s}{'Potential(H)':>15s}{'Total(H)':>13s}{'Temperature(K)':>17s}{'norm':>8s}"
        dynamics_step_info += INIT

        print (dynamics_step_info, flush=True)

    def print_step(self, istep, itrajectory):
        """ Routine to print each trajectory infomation at each step about dynamics

            :param integer istep: Current MD step
            :param integer itrajectory: Current trajectory
        """
        ctemp = self.mol.ekin * 2. / float(self.mol.ndof) * au_to_K
        norm = 0.
        for ist in range(self.mol.nst):
            norm += self.mol.rho.real[ist, ist]

        # Print INFO for each step
        INFO = f" INFO_{itrajectory+1}{istep + 1:>9d}"
        INFO += f"{self.mol.ekin:14.8f}{self.mol.epot:15.8f}{self.mol.etot:15.8f}"
        INFO += f"{ctemp:13.6f}"
        INFO += f"{norm:11.5f}"
        print (INFO, flush=True)
        
        # Print event in CTMQC
        for category, events in self.event.items():
            if (len(events) != 0):
                for ievent in events:
                    print (f" {category}{istep + 1:>9d}  {ievent}", flush=True)
        self.event["DECO"] = []
