"""

FENICS script for solving the thermal Biot system using mixed elements with monolithic solver

Author: Mats K. Brun

"""
#from fenics import *
#from dolfin.cpp.mesh import *
#from dolfin.cpp.io import *
from dolfin import *

import numpy as np
import sympy as sy

# <editor-fold desc="Parameters">
dim = 2                                                  # spatial dimension
eps = 10.0E-6                                            # error tolerance
T_final = 1.                                             # final time
number_of_steps = 10                                     # number of steps
dt = T_final / number_of_steps                           # time step
alpha = 1.                                               # Biot's coeff
beta = 1.                                                # thermal coeff
c0 = 1.                                                  # Biot modulus
a0 = 1.                                                  # thermal coeff2
b0 = 1.                                                  # coupling coeff
K = 1.                                                   # permeability
Th = 1.                                                  # th conductivity
lambd = 1.                                               # Lame param 1
mu = 1.                                                  # Lame param 2
cr = alpha**2/(mu + lambd)
ar = beta**2/(mu + lambd)
br = alpha*beta/(mu + lambd)
# </editor-fold>

# <editor-fold desc="Exact solutions and RHS">

# Define variables used by sympy
x0, x1, ti = sy.symbols('x[0], x[1], t')

# Exact solutions
pres = ti*x0*(1. - x0)*x1*(1. - x1)                    # pressure
temp = ti*x0*(1. - x0)*x1*(1. - x1)                    # temperature
disp1 = ti*x0*(1. - x0)*x1*(1. - x1)                   # displacement comp 1
disp2 = ti*x0*(1. - x0)*x1*(1. - x1)                   # displacement comp 2

pres_x = sy.diff(pres, x0)
pres_y = sy.diff(pres, x1)
df1 = - K*pres_x                                       # Darcy flux comp 1
df2 = - K*pres_y                                       # Darcy flux comp 2

temp_x = sy.diff(temp, x0)
temp_y = sy.diff(temp, x1)
hf1 = - Th*temp_x                                      # heat flux comp 1
hf2 = - Th*temp_y                                      # heat flux comp 2

# partial derivatives
disp1_x = sy.diff(disp1, x0)
disp1_y = sy.diff(disp1, x1)
disp1_xx = sy.diff(disp1, x0, x0)
disp1_yy = sy.diff(disp1, x1, x1)
disp1_xy = sy.diff(disp1, x0, x1)

disp2_x = sy.diff(disp2, x0)
disp2_y = sy.diff(disp2, x1)
disp2_xx = sy.diff(disp2, x0, x0)
disp2_yy = sy.diff(disp2, x1, x1)
disp2_xy = sy.diff(disp2, x0, x1)

df1_x = sy.diff(df1, x0)
df2_y = sy.diff(df2, x1)

hf1_x = sy.diff(hf1, x0)
hf2_y = sy.diff(hf2, x1)

# stress
sig11 = 2.*mu*disp1_x + lambd*(disp1_x + disp2_y) - alpha*pres - beta*temp
sig12 = mu*(disp1_y + disp2_x)
sig21 = mu*(disp2_x + disp1_y)
sig22 = 2.*mu*disp2_y + lambd*(disp1_x + disp2_y) - alpha*pres - beta*temp


# right hand sides
F1 = - 2.*mu*(disp1_xx + .5*(disp2_xy + disp1_yy)) \
    - lambd*(disp1_xx + disp2_xy) + alpha*pres_x + beta*temp_x

F2 = - 2.*mu*(disp2_yy + .5*(disp1_xy + disp2_xx)) \
    - lambd*(disp1_xy + disp2_yy) + alpha*pres_y + beta*temp_y

h = sy.diff(c0*pres - b0*temp + alpha*(disp1_x + disp2_y), ti) + df1_x + df2_y

f = sy.diff(a0*temp - b0*pres + beta*(disp1_x + disp2_y), ti) + hf1_x + hf2_y

# simplify expressions
pres = sy.simplify(pres)
temp = sy.simplify(temp)
disp1 = sy.simplify(disp1)
disp2 = sy.simplify(disp2)
df1 = sy.simplify(df1)
df2 = sy.simplify(df2)
hf1 = sy.simplify(hf1)
hf2 = sy.simplify(hf2)
sig11 = sy.simplify(sig11)
sig12 = sy.simplify(sig12)
sig21 = sy.simplify(sig21)
sig22 = sy.simplify(sig22)
F1 = sy.simplify(F1)
F2 = sy.simplify(F2)
h = sy.simplify(h)
f = sy.simplify(f)

# convert expressions to C++ syntax
pres_cc = sy.printing.ccode(pres)
temp_cc = sy.printing.ccode(temp)
disp1_cc = sy.printing.ccode(disp1)
disp2_cc = sy.printing.ccode(disp2)
df1_cc = sy.printing.ccode(df1)
df2_cc = sy.printing.ccode(df2)
hf1_cc = sy.printing.ccode(hf1)
hf2_cc = sy.printing.ccode(hf2)
sig11_cc = sy.printing.ccode(sig11)
sig12_cc = sy.printing.ccode(sig12)
sig21_cc = sy.printing.ccode(sig21)
sig22_cc = sy.printing.ccode(sig22)
F1_cc = sy.printing.ccode(F1)
F2_cc = sy.printing.ccode(F2)
h_cc = sy.printing.ccode(h)
f_cc = sy.printing.ccode(f)

# print the exact solutions and RHS
print """ Exact solutions as ccode:
p = \t %r
T = \t %r
u1 = \t %r
u2 = \t %r
w1 = \t %r
w2 = \t %r
r1 = \t %r
r2 = \t %r
F1 = \t %r
F2 = \t %r
h = \t %r
f = \t %r
""" % (pres_cc, temp_cc, disp1_cc, disp2_cc, df1_cc, df2_cc, hf1_cc, hf2_cc,
       F1_cc, F2_cc, h_cc, f_cc)

# </editor-fold>

# <editor-fold desc="Mesh and function spaces">
# generate unit square mesh
mesh = UnitSquareMesh(4, 4)
mesh_size = mesh.hmax()

# finite element space
DGxDG = VectorElement('DG', mesh.ufl_cell(), 0)                         # displacement
DG = FiniteElement('DG', mesh.ufl_cell(), 0)                            # pres and temp
BDMxBDM = VectorElement('BDM', mesh.ufl_cell(), 1)                      # stress
RT = FiniteElement('RT', mesh.ufl_cell(), 1)                            # fluxes
# mixed space
X = FunctionSpace(mesh, MixedElement(DGxDG, BDMxBDM, DG, RT, DG, RT, DG))

# exact solutions and RHS
p_ex = Expression(pres_cc, degree=5, t=0)
T_ex = Expression(temp_cc, degree=5, t=0)
u_ex = Expression((disp1_cc, disp2_cc), degree=5, t=0)
w_ex = Expression((df1_cc, df2_cc), degree=5, t=0)
r_ex = Expression((hf1_cc, hf2_cc), degree=5, t=0)
sig_ex = Expression(((sig11_cc, sig12_cc),
                     (sig21_cc, sig22_cc)), degree=5, t=0)

F = Expression((F1_cc, F2_cc), degree=1, t=0)
h = Expression(h_cc, degree=1, t=0)
f = Expression(f_cc, degree=1, t=0)

#F = Constant((0.0, 1.0))
#h = Constant(1.0)
#f = Constant(1.0)
# </editor-fold>


# <editor-fold desc="BC and IC">
# Define boundary points
def boundary(x, on_boundary):
    return on_boundary

# Dirichlet BC for displacement and pressure
#bc_su = DirichletBC(X.sub(0), Constant((0.0, 0.0)), boundary)
#bc_wp = DirichletBC(X.sub(2), Constant(0.0), boundary)
#bc_rt = DirichletBC(X.sub(4), Constant(0.0), boundary)
#bcs = [bc_su, bc_wp, bc_rt]

# trial and test functions
v, tau, q, z, S, y, e = TestFunctions(X)
u, sig, p, w, T, r, x = TrialFunctions(X)


# initial conditions (homogenous) and previous time-step
mf_n = Function(X)

u_n, sig_n, p_n, w_n, T_n, r_n, x_n = split(mf_n)

# </editor-fold>


# <editor-fold desc="Variational form">
# ID matrix
Id = Identity(dim)

# define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))
    #return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# define compliance tensor
def compl(s):
    return 1/(2*mu)*(s - lambd/2/(mu + lambd)*tr(s)*Id)

# skew matrix determined by scalar
def skw(r):
    return as_matrix([[0, r], [-r, 0]])

# Constants for use in var form
#dt = Constant(dt)
#alpha = Constant(alpha)
#beta = Constant(beta)
#c0 = Constant(c0)
#a0 = Constant(a0)
#cr = Constant(cr)
#ar = Constant(ar)
#b0 = Constant(b0)
#br = Constant(br)
#K = Constant(K)
#Th = Constant(Th)
#lambd = Constant(lambd)
#mu = Constant(mu)

A1 = inner(compl(sig),tau)*dx + dot(u, div(tau))*dx \
    + cr/(2*alpha)*p*tr(tau)*dx + ar/(2*beta)*T*tr(tau)*dx

A2 =  - dot(div(sig), v)*dx

A3 = (c0 + cr)*p*q*dx - (b0 - br)*T*q*dx + cr/(2*alpha)*tr(sig)*q*dx + dt*div(w)*q*dx

A4 = 1/K*dot(w,z)*dx - p*div(z)*dx

A5 = (a0 + ar)*T*S*dx - (b0 - br)*p*S*dx + ar/(2*beta)*tr(sig)*S*dx + dt*div(r)*S*dx

A6 = 1/Th*dot(r,y)*dx - T*div(y)*dx

A7 = inner(skw(x), tau)*dx + inner(sig, skw(e))*dx

L1 = dot(F,v)*dx + h*q*dx + f*S*dx

L2 = (c0 + cr)*p_n*q*dx - (b0 - br)*T_n*q*dx + cr/(2*alpha)*tr(sig_n)*q*dx \
    + (a0 + ar)*T_n*S*dx - (b0 - br)*p_n*S*dx + ar/(2*beta)*tr(sig_n)*S*dx

A = dt*A1 + dt*A2 + A3 + dt*A4 + A5 + dt*A6 + dt*A7
L = dt*L1 + L2

mf = Function(X)

# </editor-fold>

# Create VTK file for saving solution, .pvd or .xdmf
vtkfile_u = File('ThBiot_monolithic/displacement.pvd')
vtkfile_s = File('ThBiot_monolithic/stress.pvd')
vtkfile_p = File('ThBiot_monolithic/pressure.pvd')
vtkfile_w = File('ThBiot_monolithic/darcyFlux.pvd')
vtkfile_T = File('ThBiot_monolithic/temp.pvd')
vtkfile_r = File('ThBiot_monolithic/energyFlux.pvd')


# initialize time
t = 0.0

# start computation
for i in range(number_of_steps):
    # update time
    t += float(dt)
    p_ex.t = t
    T_ex.t = t
    u_ex.t = t
    w_ex.t = t
    r_ex.t = t
    sig_ex.t = t
    F.t = t
    h.t = t
    f.t = t


    # solve linear system
    #solve(F == 0, wpu, bcs)
    #solve(A == L, mf, bcs)
    solve(A == L, mf)
    _u_, _sig_, _p_, _w_, _T_, _r_, _x_ = mf.split()

    # update previous time step
    mf_n.assign(mf)

    # Compute errors in L2 norm
    p_L2 = errornorm(p_ex, _p_, 'L2')
    T_L2 = errornorm(T_ex, _T_, 'L2')
    u_L2 = errornorm(u_ex, _u_, 'L2')
    w_L2 = errornorm(w_ex, _w_, 'L2')
    r_L2 = errornorm(r_ex, _r_, 'L2')
    sig_L2 = errornorm(sig_ex, _sig_, 'L2')

    # print errors
    print """ \n Errors in L2 norm: \n
    \t Pressure: \t \t %r \n
    \t Temperature: \t \t %r \n
    \t Displacement: \t \t %r \n
    \t Darcy flux: \t \t %r \n
    \t Heat flux: \t \t %r \n
    \t Stress: \t \t %r
    """ % (p_L2, T_L2, u_L2, w_L2, r_L2, sig_L2)

    # save to file
    vtkfile_u << _u_, t
    vtkfile_s << _sig_, t
    vtkfile_p << _p_, t
    vtkfile_w << _w_, t
    vtkfile_T << _T_, t
    vtkfile_r << _r_, t



# print mesh size
print """ \n Mesh size: \n
\t %r \n
Time step: \n
\t %r 
""" % (mesh_size, dt)
