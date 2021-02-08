
from pyscf.pbc import scf, dft, gto, grad
import numpy as np
from pyscf.lib import logger

def gen_cell(atom):
    cell = gto.Cell()
    cell.atom = atom
    cell.basis = 'gth-tzvp'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.exp_to_discard=0.1
    cell.precision=1e-12
    cell.build()
    return cell

if __name__ == '__main__':
    disp = 1e-4
    # this is a cell slightly displaced from equlibrium. 
    atom_0 =  [["C", [0.000000000000, 0.000000000000, 0.001000000000]],
               ["C", [1.685068664391, 1.685068664391, 1.685068664391]]]

    cell = gen_cell(atom_0)
    kpts = cell.make_kpts([1,1,1])
    mf = dft.KRKS(cell, kpts)
    mf.xc = 'pbe,pbe'
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    # computing analytical force
    force = mf.nuc_grad_method().kernel()

    # displace +/- in the z direction on the 1st C
    atom_c0_plus = [["C", [0.000000000000, 0.000000000000, 0.001000000000+disp/2.0]],
                   ["C", [1.685068664391, 1.685068664391, 1.685068664391]]]

    atom_c0_min = [["C", [0.000000000000, 0.000000000000, 0.001000000000-disp/2.0]],
                   ["C", [1.685068664391, 1.685068664391, 1.685068664391]]]

    # compute energy for these cells with displacement on 1st C
    cell_c0_plus = gen_cell(atom_c0_plus)
    mf = dft.KRKS(cell_c0_plus, kpts)
    mf.xc = 'pbe,pbe'
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    e_c0_plus = mf.kernel()

    cell_c0_min = gen_cell(atom_c0_min)
    mf = dft.KRKS(cell_c0_min, kpts)
    mf.xc = 'pbe,pbe'
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    e_c0_min = mf.kernel()
    
    # finite differnce force on 1st C
    force_c0 = (e_c0_plus-e_c0_min) / disp

    # displace +/- in the z direction on the 2nd C
    atom_c1_plus = [["C", [0.000000000000, 0.000000000000, 0.001000000000]],
                   ["C", [1.685068664391, 1.685068664391, 1.685068664391+disp/2.0]]]

    atom_c1_min = [["C", [0.000000000000, 0.000000000000, 0.001000000000]],
                   ["C", [1.685068664391, 1.685068664391, 1.685068664391-disp/2.0]]]
    
    # compute energy for these cells with displacement on 2nd C
    cell_c1_plus = gen_cell(atom_c1_plus)
    mf = dft.KRKS(cell_c1_plus, kpts)
    mf.xc = 'pbe,pbe'
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    e_c1_plus = mf.kernel()

    cell_c1_min = gen_cell(atom_c1_min)
    mf = dft.KRKS(cell_c1_min, kpts)
    mf.xc = 'pbe,pbe'
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    e_c1_min = mf.kernel()
    
    # finite differnce force on 2nd C
    force_c1 = (e_c1_plus-e_c1_min) / disp
    print("Analytical force:")
    print(force)
    print("===========")
    print("finite difference",force_c0, force_c1)
    print("difference with analytical gradients:")
    print(abs(force_c0-force[0][2]), abs(force_c1-force[1][2]))
    print("net force within the cell:", force_c0+force_c1)
