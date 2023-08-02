# numpy for fast arrays and some linear algebra.
import numpy as np
# import matplotlib for plotting
import matplotlib.pyplot as plt
# import Dormand-Prince Butcher tableau
import dormand_prince as dp
# import our Runge-Kutta integrators
import runge_kutta as rk

def dormand_prince_integrator(f, x, t, h):
    a = dp.a
    b = dp.b
    c = dp.c
    # print ("in dormand: ",x)
    x = rk.explicit_RK_stepper(f,x,t,h,a,b,c)
    return x
    # return a # please complete this function
               # so it returns the prediction for the
               # Dormand-Prince method
               # To that end, use rk.explicit_rk_stepper!
               # return x
# Feel free play around with the following quantities
# and see how the solution changes!

# time horizon
tspan = (0.0,2.0)
# time step
h = 0.2
# initial condition
x_0 = 3.0

########################################
### hereafter no more code modification necessary
########################################

# model right-hand-side
def f(x,t):
    # print (x)
    return -2*x

# simulate model
# dormand_prince_integrator(f, x_0, tspan[0], h)
trajectory, time_points = rk.integrate(f, # ODE right-hand-side
                                         x_0, # initial condition
                                         tspan, # time horizon
                                         h, # time step
                                         dormand_prince_integrator) # integrator

# analytical solution
time_points_analytical = np.linspace(tspan[0],tspan[1], 1000)
trajectory_analytical = x_0*np.exp(-2*time_points_analytical)

# # plot trace
fig, ax = plt.subplots()
ax.set_xlabel("time")
ax.set_ylabel("x(t)")

ax.plot(time_points, trajectory, linewidth=2, color="red", marker = "o", label ="Analytical")
ax.plot(time_points_analytical, trajectory_analytical, linewidth=2, color="black", linestyle="dashed", label = "Numerical ")
ax.legend()
plt.show()
