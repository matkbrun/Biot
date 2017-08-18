"""

FENICS script for solving the Biot system using iterative fixed stress splitting method w
with mixed elements

Author: Mats K. Brun

"""
from fenics import *
from dolfin.cpp.mesh import *
from dolfin.cpp.io import *

import numpy as np
import sympy as sy

# <editor-fold desc="Parameters">

dim = 2                                                  # spatial dimension
eps = 10.0E-6                                            # error tolerance
T_final = 1.0                                            # final time
number_of_steps = 10                                     # number of steps
dt = T_final / number_of_steps                           # time step
alpha = 1.0                                              # Biot's coeff
E = 1.0                                                  # bulk modulus
nu = 0.25                                                # Poisson ratio \in (0, 0.5)
M = 1.0                                                  # Biot modulus
K = 1.0                                                  # permeability divided by fluid viscosity
lambd = 3.0*E*nu/(nu + 1.0)                              # Lame param 1
mu = 3.0*E*(1.0 - 2.0*nu)/2.0/(nu + 1.0)                 # Lame param 2
betaFS = alpha**2/2.0/(2.0*mu/dim + lambd)               # tuning param
# </editor-fold>

# <editor-fold desc="Exact solutions and RHS">

# Define variables used by sympy
x, y, t = sy.symbols('x[0], x[1], t')

# Exact solutions
pressure = t*x*(1.0 - x)*y*(1.0 - y)                    # pressure
u1 = t*x*(1.0 - x)*y*(1.0 - y)                          # displacement comp 1
u2 = t*x*(1.0 - x)*y*(1.0 - y)                          # displacement comp 2

px = sy.diff(pressure, x)
py = sy.diff(pressure, y)
w1 = - K*px                                              # flux comp 1
w2 = - K*py                                              # flux comp 2

# partial derivatives
u1x = sy.diff(u1, x)
u1y = sy.diff(u1, y)
u1xx = sy.diff(u1, x, x)
u1yy = sy.diff(u1, y, y)
u1xy = sy.diff(u1, x, y)

u2x = sy.diff(u2, x)
u2y = sy.diff(u2, y)
u2xx = sy.diff(u2, x, x)
u2yy = sy.diff(u2, y, y)
u2xy = sy.diff(u2, x, y)

w1x = sy.diff(w1, x)
w2y = sy.diff(w2, y)


# right hand sides
Sf = sy.diff(1.0/M*pressure + alpha*(u1x + u2y), t) + w1x + w2y

f1 = - 2.0*mu*(u1xx + 0.5*(u2xy + u1yy)) - lambd*u1xx + alpha*px

f2 = - 2.0*mu*(u2yy + 0.5*(u1xy + u2yy)) - lambd*u2yy + alpha*py

# simplify expressions
pressure = sy.simplify(pressure)
u1 = sy.simplify(u1)
u2 = sy.simplify(u2)
w1 = sy.simplify(w1)
w2 = sy.simplify(w2)
Sf = sy.simplify(Sf)
f1 = sy.simplify(f1)
f2 = sy.simplify(f2)

# convert expressions to C++ syntax
p_code = sy.printing.ccode(pressure)
u1_code = sy.printing.ccode(u1)
u2_code = sy.printing.ccode(u2)
w1_code = sy.printing.ccode(w1)
w2_code = sy.printing.ccode(w2)
Sf_code = sy.printing.ccode(Sf)
f1_code = sy.printing.ccode(f1)
f2_code = sy.printing.ccode(f2)

# print the exact solutions and RHS
print """ Exact solutions as ccode:
p = \t %r
u1 = \t %r
u2 = \t %r
w1 = \t %r
w2 = \t %r
Sf = \t %r
f1 = \t %r
f2 = \t %r
""" % (p_code, u1_code, u2_code, w1_code, w2_code, Sf_code, f1_code, f2_code)

# </editor-fold>


# <editor-fold desc="Mesh and function spaces">
# generate unit square mesh
mesh = UnitSquareMesh(128, 128)
mesh_size = mesh.hmax()

# define function spaces
U_elem = VectorElement('P', mesh.ufl_cell(), 1)         # space for displacement
P_elem = FiniteElement('DG', mesh.ufl_cell(), 0)        # element for pressure
W_elem = FiniteElement('RT', mesh.ufl_cell(), 1)        # element for flux
elem = MixedElement([W_elem, P_elem, U_elem])           # mixed element
WPU = FunctionSpace(mesh, elem)                         # mixed space

# exact solutions and RHS
w_ex = Expression((w1_code, w2_code), degree=5, t=0)
p_ex = Expression(p_code, degree=5, t=0)
u_ex = Expression((u1_code, u2_code), degree=5, t=0)
Sf = Expression(Sf_code, degree=1, t=0)
f = Expression((f1_code, f2_code), degree=1, t=0)
# </editor-fold>


# <editor-fold desc="BC and IC">
# Define boundary points
def boundary(x, on_boundary):
    return on_boundary

# Dirichlet BC for displacement and pressure
bc_wp = DirichletBC(WPU.sub(1), p_ex, boundary)
bc_u = DirichletBC(WPU.sub(2), u_ex, boundary)
bcs = [bc_wp, bc_u]

# trial and test functions
z, q, v = TestFunctions(WPU)
w, p, u = TrialFunctions(WPU)


# initial conditions (homogenous) and previous time-step/iteration
#wpu = Function(WPU)
wpu_n = Function(WPU)

#w, p, u = split(wpu)
w_n, p_n, u_n = split(wpu_n)

# </editor-fold>


# <editor-fold desc="Variational form">
# define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Constants for use in var form
dt = Constant(dt)
alpha = Constant(alpha)
M = Constant(M)
K = Constant(K)
g = Constant((0.0, 0.0))    # gravitational force
lambd = Constant(lambd)
mu = Constant(mu)
betaFS = Constant(betaFS)
rho_f = Constant(1.0)       # fluid density

# define var problem
#F = 1/M*p*q*dx + alpha*div(u)*q*dx + dt*div(w)*q*dx + 1/K*dot(w, z)*dx - p*div(z)*dx \
#     - dt*Sf*q*dx - 1/M*p_n*q*dx - alpha*div(u_n)*q*dx - rho_f*dot(g, z)*dx \
#     + 2*mu*inner(epsilon(u), epsilon(v))*dx + lambd*div(u)*div(v)*dx - alpha*p*div(v)*dx - dot(f, v)*dx

a = 1/M*p*q*dx + alpha*div(u)*q*dx + dt*div(w)*q*dx + 1/K*dot(w, z)*dx - p*div(z)*dx \
    + 2*mu*inner(epsilon(u), epsilon(v))*dx + lambd*div(u)*div(v)*dx - alpha*p*div(v)*dx

L = dt*Sf*q*dx + 1/M*p_n*q*dx + alpha*div(u_n)*q*dx + rho_f*dot(g, z)*dx + dot(f, v)*dx

wpu = Function(WPU)

# </editor-fold>

# Create VTK file for saving solution, .pvd or .xdmf
vtkfile_w = File('Biot_monolithic/flux.pvd')
vtkfile_p = File('Biot_monolithic/pressure.pvd')
vtkfile_u = File('Biot_monolithic/displacement.pvd')

# initialize time
t = 0.0

# start computation
for i in range(number_of_steps):
    # update time
    t += float(dt)
    u_ex.t = t
    w_ex.t = t
    p_ex.t = t
    Sf.t = t
    f.t = t

    # solve linear system
    #solve(F == 0, wpu, bcs)
    solve(a == L, wpu, bcs)
    _w_, _p_, _u_ = wpu.split()

    # update previous time step
    wpu_n.assign(wpu)

    # <editor-fold desc="Compute and print errors">
    # Compute errors in L2 norm
    flux_error_L2 = errornorm(w_ex, _w_, 'L2')
    pressure_error_L2 = errornorm(p_ex, _p_, 'L2')
    displacement_error_L2 = errornorm(u_ex, _u_, 'L2')

    # interpolate exact solutions at current step
    w_e = interpolate(w_ex, WPU.sub(0).collapse())
    p_e = interpolate(p_ex, WPU.sub(1).collapse())
    u_e = interpolate(u_ex, WPU.sub(2).collapse())

    # Compute maximum error at vertices
    vertex_values_w_e = w_e.compute_vertex_values(mesh)
    vertex_values_w = _w_.compute_vertex_values(mesh)
    flux_error_max = np.max(np.abs(vertex_values_w_e - vertex_values_w))

    vertex_values_u_e = u_e.compute_vertex_values(mesh)
    vertex_values_u = _u_.compute_vertex_values(mesh)
    displacement_error_max = np.max(np.abs(vertex_values_u_e - vertex_values_u))

    vertex_values_p_e = p_e.compute_vertex_values(mesh)
    vertex_values_p = _p_.compute_vertex_values(mesh)
    pressure_error_max = np.max(np.abs(vertex_values_p_e - vertex_values_p))

    # print errors
    print """ \n Errors in L2 and max norm: \n
    \t flux error in L2-norm: \t \t %r
    \t flux error in max-norm: \t \t %r \n
    \t pressure error in L2-norm: \t \t %r
    \t pressure error in max-norm: \t \t %r \n
    \t displacement error in L2-norm: \t %r
    \t displacement error in max-norm: \t %r
    """ % (flux_error_L2, flux_error_max, pressure_error_L2,
           pressure_error_max, displacement_error_L2, displacement_error_max)
    # </editor-fold>

    # save to file
    vtkfile_w << _w_, t
    vtkfile_u << _u_, t
    vtkfile_p << _p_, t

# print value of parameters
print """ Parameters: \n
\t time step: \t %r
\t final time: \t %r
\t lambda: \t %r
\t mu: \t \t %r
\t betaFS: \t %r \n """ % (float(dt), float(t), float(lambd), float(mu), float(betaFS))

# print mesh size
print """ Mesh size: \n
\t %r
""" % mesh_size
