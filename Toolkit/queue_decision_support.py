##################################################################### Network Representations for Use-Cases ##################################################################
import math

def get_approx_measures(l,mu,c):

    '''
    Generates approximate measures to be used in the SBC approach:

    Args:
        l (np.array) : arrival rates per unit of time.
        mu (np.array) : processing rate of server per unit of time.
        c (np.array) : available servers per unit of time.

    Returns:
        P (float): Probability of the system having a backlog (not serving everyone at this timestep)
        E (float): Utilization of the servers. (probability of the system being empty)
        l_mar (float) : Modified arrival rate for the SBC approach
    '''

    P = (l/mu)**c/(math.factorial(c)*(sum([((l/mu)**x)/(math.factorial(x)) for x in range(0,c+1)])))
    E = (l-P*l)/(c*mu)
    l_mar = c*mu*E
    return P,E,l_mar

def get_mmc_measures(l_mar,mu,c):

    '''
    Generates approximate measures from the M/M/C queue using the formulas from
    Queueing Networks And Markov Chains - Modelling And Performance Evaluation With Computer Science Applications (p.218,Eq.6.27-28):

    Args:
        l_mar (np.array) : modified arrival rates per unit of time.
        mu (np.array) : processing rate of server per unit of time.
        c (np.array) : available servers per unit of time.

    Returns:
        L_q (float): Number of people in the queue
        W (float): Waiting time in the queue in minutes
    '''

    rho = l_mar/(c*mu)
    cons = ((c*rho)**c)/(math.factorial(c)*((1-rho)))
    P_0 = sum([((c*rho)**x)/(math.factorial(x))  for x in range(0,c)]) + cons
    P_0 = (P_0)**(-1)
    Lq = (P_0 *(l_mar/mu)**c*rho)/(math.factorial(c)*((1-rho)**2))

    if l_mar == 0 :
        l_mar = 1e-16
    W = Lq/l_mar
    return Lq,W

def get_mdc_measures(l_mar,mu,c,W_m):

    '''
    Generates approximate measures from the M/D/C queue using the formulas from
    Queueing Networks And Markov Chains - Modelling And Performance Evaluation With Computer Science Applications (p.231,Eq.6.82):

    Args:
        l_mar (np.array) : modified arrival rates per unit of time.
        mu (np.array) : processing rate of server per unit of time.
        c (np.array) : available servers per unit of time.

    Returns:
        L_q (float): Number of people in the queue
        W (float): Waiting time in the queue in minutes

    '''

    rho = l_mar/(c*mu)
    if rho == 0 :
        rho = 1e-16
    nc = (1 + (1-rho)*(c-1)*((4+5*c)**0.5-2)/(16*rho*c))**(-1)
    W = 0.5*(1/(nc))*W_m
    Lq = W*l_mar
    return Lq,W
