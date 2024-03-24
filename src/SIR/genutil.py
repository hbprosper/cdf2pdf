import numpy as np
from scipy.integrate import solve_ivp

class FluData:
    
    def __str__(self):
        s = ''
        for field in dir(self):
            if field[0] == '_': continue
            value = self.__getattribute__(field)
            s += ' %-12s: %s\n' % (field, value)
        return s
    
class EbolaData:
    
    def __str__(self):
        s = ''
        for field in dir(self):
            if field[0] == '_': continue
            value = self.__getattribute__(field)
            s += ' %-12s: %s\n' % (field, value)
        return s


SEIRdata= EbolaData()
SEIRdata.model = 'SEIR'

SEIRdata.alpha0 = 0 ##
SEIRdata.beta0  = 0 ##
SEIRdata.eta0   = 0 #

SIRdata = FluData()
SIRdata.model = 'SIR'

# Dr. M's parameters
SIRdata.alpha0 = 0.46500
SIRdata.beta0  = 0.00237

# initial state
SIRdata.S0     = 763  # initial number of susceptible persons
SIRdata.I0     =   3  # initial number of infected
SIRdata.R0     =   0  # initial number of recovered persons 

SIRdata.alpha_scale= 1.0
SIRdata.alpha_bins = 16
SIRdata.alpha_min  = 0.0
SIRdata.alpha_max  = 1.0

SIRdata.beta_scale =  5.0e-3
SIRdata.beta_bins  = 16
SIRdata.beta_min   =  0.2
SIRdata.beta_max   =  0.7

SIRdata.tmin       =  0.0
SIRdata.tmax       = 14.0

SIRdata.scale      = 50 # scale for counts and statistics

# Boarding School Data
SIRdata.T = np.array([0,  2,  3,   4,   5,   6,   7,   8,   
                      9, 10, 11, 12, 13])

SIRdata.D = np.array([3, 25, 75, 227, 296, 258, 236, 192, 
                      126, 71, 28, 11,  7])
SIRdata.O = SIRdata.D

def generate(params, data,
             maxiter=1000000):
    '''
    Generate data from a single simulated epidemic.
    
    INPUTS
        params: parameter point of model. if SIR, alpha and beta 
                     will be rescaled as follows:
                       alpha *= data.alpha_scale
                       beta  *= data.beta_scale
 
        data:        object containing: 
                        T, S0, I0, R0, alpha_scale, beta_scale, 
                        tmin, tmax
    
    RETURN
        s, i, r:     np arrays of counts
    '''

    
    if data.model == 'SIR':

        # scale alpha, beta to correct scale
        alpha, beta = params
        alpha *= data.alpha_scale
        beta  *= data.beta_scale
        s0     = data.S0
        i0     = data.I0
        r0     = data.R0
        transition = [(1, 0), # infection
                      (0, 1)] # removal 
        
    elif data.model == 'SEIR':
        
        # scale alpha, beta to correct scale
        alpha, beta, eta = params
        alpha *= data.alpha_scale # x *= a same as x = x*a, x -=, x +=
        beta  *= data.beta_scale
        eta   *= data.eta_scale
        s0     = data.S0
        e0     = data.E0
        i0     = data.I0
        r0     = data.R0
    
    tmin   = data.tmin
    tmax   = data.tmax
    
    # start time
    t  = tmin

    # starting state
    st = s0
    it = i0
    rt = r0

    # initialize list of states
    states = [[t, st, it, rt]]

    ii = 0
    while (t < tmax) and (ii < maxiter):
        ii += 1
    
        # generate time to next event
        p1   = beta * it * st
        p2   = alpha * it
        psum = p1 + p2

        if psum > 0:
            t += np.random.exponential(1.0/psum)
        else:
            # since it = 0, the epidemic has ended. So make end state the
            # same as last state.
            t = 1.01 * tmax
            state = states[-1][1:] # skip time stamp
            state.insert(0, t)     # insert time at position 0
            states.append(state)
            break
            
        # choose event
        k  = np.random.choice([0, 1], p=[p1/psum, p2/psum])
        i_new, r_new = transition[k]
        
        # update state
        st = st - i_new
        it = it + i_new - r_new
        rt = rt + r_new

        # the counts should never be negative
        assert(st >= 0)
        assert(it >= 0)
        assert(rt >= 0)
        
        states.append([t, st, it, rt])
    
    return states

def observe(T, states):

    # implement a braided loop: one strand is over the epidemic events 
    # and another strand is over the observation times.

    results = []
    j = 0
    # loop over all states except the last
    for i, state in enumerate(states[:-1]):

        t = state[0]
        
        # loop over all observation times
        while j < len(T):

            # if t <= T[j] < t_next then
            #    i)  the state at time t is the same as that observed at time T[j].
            #    ii) therefore, move to the next observation time T[j] => T[j+1]
            # otherwise keep the same observation time, but go to the next epidemic state 
            if T[j] >= t:
                t_next = states[i+1][0]
                if T[j] < t_next:
                    results.append(state[1:])
                    j += 1
                else:
                    break # the observation time T[j] does not lie in the interval [t, t_next)
            else:
                break # the observation time T[j] does not lie in the interval [t, t_next)

    assert(len(results) == len(T))
    return results
    
def Fsolve(alpha, beta, data=SIRdata):
    '''
    Solve SIR model ODEs for given parameter point (alpha, beta).
    
    INPUTS
    
        alpha, beta: parameter point of SIR model. alpha and beta 
                     will be rescaled as follows:
                       alpha *= data.alpha_scale
                       beta  *= data.beta_scale
 
        data:        object containing: 
                        T, S0, I0, R0, alpha_scale, beta_scale, 
                        tmin, tmax
    
    RETURN
        soln.y:    solution y[0] = S, y[1] = I
    '''
    y0     = np.array([data.S0, data.I0])
    teval  = data.T
    alpha *= data.alpha_scale
    beta  *= data.beta_scale
    tspan  = (data.tmin, data.tmax) # time span
    
    def F(t, y):
        S  = y[0]
        I  = y[1]
        f1 =-beta*S*I
        f2 = beta*S*I - alpha*I
        return np.array([f1, f2])

    return solve_ivp(F, tspan, y0, t_eval=teval)


def test_statistic(alpha, beta, data=SIRdata, scale=True):
    '''
    Compute test statistic(s) for given data data.O and parameter
    point (alpha, beta). If alpha and beta are arrays, then a
    test statistic is computed for each point.
    
    INPUTS
        alpha, beta: parameter point of SIR model. alpha and beta 
                     will be rescaled as follows:
                       alpha *= data.alpha_scale
                       beta  *= data.beta_scale
 
        data:        object containing: 
                        T, S0, I0, R0, alpha_scale, beta_scale, 
                        tmin, tmax    
    RETURN
        test statistic(s)
    '''
    try:
        l0 = np.zeros(len(alpha))
        for j, (a, b) in enumerate(zip(alpha, beta)):
            soln = Fsolve(a, b, data)
            I = soln.y[1]
            l = data.O - I
            l = l*l / I             
            l0[j] = l.mean()
    except:
        soln  = Fsolve(alpha, beta, data)
        I = soln.y[1]
        l = data.O - I
        l = l*l / I
        l0= l.mean()
        
    l0 = np.sqrt(l0)
    
    if scale:
        return l0 / data.scale
    else:
        return l0

class TimeLeft:
    '''
    Return the amount of time left.
    
    timeleft = TimeLeft(N)
    
    N: maximum loop count
    
    for i in range(N):
        s = timeleft(i)
    '''
    def __init__(self, N, step=10):
        import time

        self.N     = N
        self.step  = step
        self.ii    = 0
        
        self.timenow = time.time
        self.start   = self.timenow()
        
    def __del__(self):
        pass
    
    def string(self):
        
        # elapsed time since start
        s = self.timenow() - self.start
        # time/loop
        loop = self.ii + 1
        t = s / loop
        
        # time left
        s = t * (self.N - loop)
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s =  s - 60*m
        percent = 100 * loop / self.N
        
        rec = "%10d | %6.2f%s | %2.2d:%2.2d:%2.2d" % \
        (self.ii, percent, '%', h, m, s)
        
        self.ii += self.step
        return rec
    
    def __call__(self):
        if self.ii % self.step == 0:
            print(f'\r{self.string():s}', end='')