# numpy for fast arrays and some linear algebra.
import numpy as np
# import matplotlib for plotting
import matplotlib.pyplot as plt
# import Dormand-Prince Butcher tableau
import dormand_prince as dp
# import our Runge-Kutta integrators
import runge_kutta as rk

# Please implement here:
#    S - numpy array carrying the stoichiometry matrix
#    k - numpy array carrying the rate coefficients k1 = 1, k2 = 2, k3=1
#    c_0 - initial composition, i.e., c_0(X) = 1.0, c_0(Y)= 0.25, C_0(P) = 0.0



# reaction network
# A+X --> 2X
# X + Y --> 2Y
# Y --> P


def reaction_rates(c,k):
    """
        Function implementing the reaction rate computation of our toy reactor

        inputs:
            c - concentration of species X, Y, P (numpy array)
            k - rate constants (organized as list)

        outputs:
            reaction rates (numpy array)
    """
    r = [k[0]*c[0], k[1]*c[1]*c[0], k[2]*c[1]]
    return np.array(r) # please complete according to the given reaction network

def reactor(c,t,k,S):
    """
        Function returing the rhs of our toy reactor model

        inputs:
            c - concentration of species  (numpy array)
            t - time
            k - rate constants (organized as list)
            S - stoichiometry matrix (numpy array)

        outputs:
            dc/dt - numpy array
    """
    r_values = reaction_rates(c,k)
    dcdt = np.dot(S,r_values)

    return dcdt # please complete this function

# Please play around with the step size to study the effect on the solution
h = 1e-2

########################################
### hereafter no more code modification necessary
########################################

# time horizon
tspan = (0.0,50.0)


# define dormant_prince_stepper
def dormant_prince_stepper(f,x,t,h):
    return rk.explicit_RK_stepper(f,x,t,h,dp.a,dp.b,dp.c)


S = np.array([[1, -1, 0], [0, 1, -1], [0,0,1]])
c_0 = np.array([1., 0.25, 0])
k = np.array([1, 2, 1])
trajectory, time_points = rk.integrate(lambda c, t: reactor(c, t, k, S),
                                       c_0,
                                       tspan,
                                       h,
                                       dormant_prince_stepper)

species_names = ["X", "Y", "P"]
colors = ["red", "blue", "black"]

fig, ax = plt.subplots()
ax.set_xlabel("time")
ax.set_ylabel("concentration")
for i in range(3):
    ax.plot(time_points, [c[i] for c in trajectory],
            color=colors[i],
            linewidth=2,
            label = species_names[i])
ax.legend(loc="center right")
fig.savefig("bonus_concentration_traces.pdf")
