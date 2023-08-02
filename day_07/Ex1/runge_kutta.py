from numpy.linalg import norm
import numpy as np

def explicit_RK_stepper(f,x,t,h,a,b,c):
    """
        Implementation of generic explicit Runge-Kutta update for explicit ODEs

        inputs:
            x - current state
            t - current time
            f - right-hand-side of the (explicit) ODE to be integrated (signature f(x,t))
            h - step size
            a - coefficients of Runge-Kutta method (organized as list-of-list (or vector-of-vector))
            b - weights of Runge-Kutta method (list/vector)
            c - nodes of Runge-Kutta method (including 0 as first node) (list/vector)

        outputs:
            x_hat - estimate of the state at time t+h
    """
    s= len(c)
    k = []
    k.append(f(x,t))

    # for i in range(s-1):
    #     k_val = 0
    #     for j in range(len(k)):
    #         # print (i, j)
    #         k_val +=h*a[i][j]*k[j]
    #         # print (a[i][j],k[j])
    #     # print (k_val)
    #     k.append(f(x+k_val, t+h*c[i]))
    # print (k)
    #
    # x_hat = x
    # for i in range(len(k)):
    #     x_hat += h*(b[i]*k[i])


    for i in range(s-1):
        x_tilde = x+ h*sum(a[i][j]*k[j] for j in range(len(k)))
        k.append(f(x_tilde, t+h*c[i]))
    x_hat = x + h*sum(b[i]*k[i] for i in range(len(k)))

    return x_hat # please complete this function

def integrate(f, x0, tspan, h, step):
    """
        Generic integrator interface

        inputs:
            f     - rhs of ODE to be integrated (signature: dx/dt = f(x,t))
            x0    - initial condition (numpy array)
            tspan - integration horizon (t0, tf) (tuple)
            h     - step size
            step   - integrator with signature:
                        step(f,x,t,h) returns state at time t+h
                        - f rhs of ODE to be integrated
                        - x current state
                        - t current time
                        - h stepsize

        outputs:
            ts - time points visited during integration (list)
            xs - trajectory of the system (list of numpy arrays)
    """
    t, tf = tspan
    x = x0
    trajectory = [x0]
    ts = [t]
    # x_new = explicit_RK_stepper(f,x,t,h,a,b,c)
    while t < tf:
        h_eff = min(h, tf-t)
        x = step(f,x,t,h_eff)
        t = min(t+h_eff, tf)
        trajectory.append(x)
        ts.append(t)
    return trajectory, ts
