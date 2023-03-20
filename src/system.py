from src.libraries import *

totalNumParams = 0
# MSD1
# Variables used to derive equations
x1, v1, m1, c1, d1, F, t, p1, s1 = symbols('x_1, v_1, m_1, c_1, d_1, F, t, p_1, s_1')

# Derivatives
dx1_dt = v1
dv1_dt = (1/m1)*(-c1*x1 - d1*v1 + F)

# System parameters 1
m1_cons = 1.0
c1_cons = 1.0
d1_cons = 1.0
totalNumParams += 3

# MSD2
# Variables used to derive equations
x2, v2, m2, c2, d2, F, t, p2, s2 = symbols('x_2, v_2, m_2, c_2, d_2, F, t, p_2, s_2')
cc, dc = symbols('c_c, d_c')

# Derivatives
dx2_dt = v2
dv2_dt = (1/m2)*(-c2*x2 - d2*v2 - F)

# outputs
F_c = cc*(x2-x1)+dc*(v2-v1)
# Equivalent
# F_c = cc*(x2-x1)+dc*(v2-v1) = cc*x2 - cc*x1 + dc*v2 - dc*v1

# System parameters 2
m2_cons = 1.0
c2_cons = 1.0
d2_cons = 1.0
totalNumParams += 3
# system parameters coupling
cc_cons = 1.0
dc_cons = 1.0
totalNumParams += 2

# Symbolic substitution in Sys2
F_exp_num = F_c.subs(cc, cc_cons).subs(dc, dc_cons)
dx1_dt_num = dx1_dt
dv1_dt_num = dv1_dt.subs(m1, m1_cons).subs(F, F_exp_num).subs(c1, c1_cons).subs(d1, d1_cons)
dx2_dt_num = dx2_dt
dv2_dt_num = dv2_dt.subs(m2, m2_cons).subs(F, F_exp_num).subs(c2, c2_cons).subs(d2, d2_cons)

# Sensitivity analysis for coupled system

Фx1F, Фv1F, Фx2x1, Фv2x1, Фx2v1, Фv2v1= symbols('Фx1_F, Фv1_F, Фx2_{x1}, Фv2_{x1}, Фx2_{v1}, Фv2_{v1}')

dФx1F_dt = Фv1F
dФv1F_dt = diff(dv1_dt, x1)*Фx1F + diff(dv1_dt, v1)*Фv1F+ diff(dv1_dt, F)
# print(dФx1F_dt)
# print(dФv1F_dt)
# dФv1F_dt

dv2F_dt = dv2_dt.subs(F, F_c)
# dv2F_dt

dФx2x1_dt = Фv2x1
dФv2x1_dt = diff(dv2F_dt, x2)*Фx2x1 + diff(dv2F_dt, v2)*Фv2x1 + diff(dv2F_dt, x1)
# print(dФx2x1_dt)
# print(dФv2x1_dt)
# dФv2x1_dt

dФx2v1_dt = Фv2v1
dФv2v1_dt = diff(dv2F_dt, x2)*Фx2v1 + diff(dv2F_dt, v2)*Фv2v1 + diff(dv2F_dt, v1)
# print(dФx2v1_dt)
# print(dФv2v1_dt)
# dФv2v1_dt