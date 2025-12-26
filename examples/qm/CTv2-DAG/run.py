from molecule import Molecule
import qm, mqc
from misc import data
import numpy as np
import random
import os
import sys

top_path = os.path.abspath(f"./")

ntraj = 200
mass = 2000.
data[f"X1"] = mass # au
t = 1.

sigma = 2.0
x0 =  -25.
k0 =  30.  # Set k0

#sigma = 20. / k0      # Set wavepacket width
tmax = -2.5 * x0 * mass / k0        # Set nstep
nstep = int(tmax / 10 / t) * 10

# Set k0 seed for initial positions and call random positions from a normal distribution
seed = 1234
np.random.seed(seed)
pos_list = np.random.normal(loc=x0, scale=sigma, size=ntraj)
mom_list = np.random.normal(loc=k0, scale=0.5 / sigma, size=ntraj)

mols = []
istates = []
for itraj in range(ntraj):
    pos = pos_list[itraj]
    #vel = k0/mass
    vel = mom_list[itraj]/mass
    geom = f"""
    1
    comment
    X1   {pos}  {vel}
    """         
    mol = Molecule(geometry=geom, ndim=1, nstates=2, ndof=1, unit_pos='au', l_model=True)
    mols.append(mol)
    istates.append(0)


bo = qm.model.DAG(molecule=mols[0])

md = mqc.CTv2(molecules=mols, istates=istates, dt=t, nsteps=nstep, nesteps=20, \
              elec_object="coefficient", propagator="rk4", l_adj_nac=False, rho_threshold=0.01, \
              init_coefs=None, unit_dt="au", out_freq=1, verbosity=2, \
              l_crunch=True, l_dc_w_mom=True, l_traj_gaussian=False, t_cons=2, l_etot0=True, l_lap=False)

md.run(qm=bo, output_dir=top_path)

del mol, mols, md, bo
gc.collect()

