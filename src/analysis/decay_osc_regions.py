import numpy as np
import matplotlib.pyplot as plt



def solver(I, a, T, dt, theta):
    """
    solving for u' = -au

    Parameters
    ----------
    I : Initial Condition
    a : decay rate
    T : Time
    dt : Time step
    theta : Method Model, theta = 1 FE, theta = 0 BE, theta = 0.5 CN

    Returns
    -------
    u : desired function
    t : t to the desired function

    """
    dt = float(dt)
    Nt = int(round(T/dt))
    T  = Nt*dt
    u = np.zeros(Nt+1)
    t = np.linspace(0, T, Nt+1)
    
    u[0] = I
    for n in range(0,Nt):
        u[n+1] = (1-(1-theta)*a*dt)/(1+theta*dt*a)*u[n]
    return u, t


def non_physical_behavior(I, a, T, dt, theta):
    """
    Given lists/arrays a and dt, and numbers I, dt, and theta,
    make a two-dimensional contour line B=0.5, where B=1>0.5
    means oscillatory (unstable) solution, and B=0<0.5 means
    monotone solution of u'=-au.
    """
    a = np.asarray(a); dt = np.asarray(dt)  # must be arrays
    B = np.zeros((len(a), len(dt)))         # results
    for i in range(len(a)):
        for j in range(len(dt)):
            u, t = solver(I, a[i], T, dt[j], theta)
            # Does u have the right monotone decay properties?
            correct_qualitative_behavior = True
            for n in range(1, len(u)):
                if u[n] > u[n-1]:  # Not decaying?
                    correct_qualitative_behavior = False
                    break  # Jump out of loop
            B[i,j] = float(correct_qualitative_behavior)
    a_, dt_ = np.meshgrid(a, dt)  # make mesh of a and dt values
    plt.contour(a_, dt_, B, levels = [1])
    plt.grid('on')
    plt.title('theta=%g' % theta)
    plt.xlabel('a'); plt.ylabel('dt')
    plt.savefig('osc_region_theta_%s.png' % theta)
    plt.savefig('osc_region_theta_%s.pdf' % theta)

non_physical_behavior(
    I=1,
    a=np.linspace(0.01, 4, 22),
    dt=np.linspace(0.01, 4, 22),
    T=6,
    theta=0.5)
