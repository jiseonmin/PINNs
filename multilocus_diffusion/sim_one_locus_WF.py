import numpy as np
import argparse
import pickle 
parser = argparse.ArgumentParser()
parser.add_argument("N", type=int, 
help='population size')
parser.add_argument("s", type=float, 
help='selection coefficient of the allele')
parser.add_argument("f0", type=float, 
help='initial allle frequency (needs to be in [0,1] range)')
parser.add_argument("nsim", type=int, 
help='number of simulations')
args = parser.parse_args()

def simulate_WF(N, n0, s):
    # initial size of the lineage = n0
    n = n0
    t = 0
    while n > 0 and n < N:
        n = np.random.binomial(N, n*(1+s) / (N-n + n*(1+s)))
        t += 1
    return t, n

if __name__ == '__main__':
    tlist = []
    nlist = []
    
    n0 = int(args.f0 * args.N)
    nsim = 0
    while nsim < args.nsim:
        t_and_n = simulate_WF(args.N, n0, args.s)
        tlist.append(t_and_n[0])
        nlist.append(t_and_n[1])
        nsim += 1

    nlist = np.array(nlist)
    tlist = np.array(tlist)
    # convert the data to be a series of p(f=1, t) and p(f=0, t)
    print(nlist)
    t_extinct_list = tlist[np.where(nlist == 0)[0]]
    t_fix_list = tlist[np.where(nlist != 0)]
    print(t_fix_list)

    # sort t extinct and tfix, the index value of t + 1 divided by nsim is the probability of fixation at that t.
    t_extinct_list.sort()
    p_extinct_list = np.arange(1, 1+len(t_extinct_list)) / nsim
    t_fix_list.sort()
    p_fix_list = np.arange(1, len(t_fix_list) + 1) / nsim

    f_initial = np.append(np.arange(0, args.f0, args.f0 / 100), np.arange(args.f0 * 101/100, 1, args.f0/100))
    f_initial = np.append(np.array([args.f0]), f_initial)
    p_initial = np.zeros(len(f_initial) - 1)
    p_initial = np.append(np.array([1]), p_initial)    
    data_dict = {'f':np.concatenate((np.zeros(len(t_extinct_list)), np.ones(len(t_fix_list)), f_initial), axis=None), 
    't':np.concatenate((t_extinct_list, t_fix_list, np.zeros(len(f_initial))), axis=None), 
    'phi':np.concatenate((p_extinct_list, p_fix_list, p_initial), axis=None)}

    np.save("single_locus_data.npy", data_dict)
    print(data_dict)

    print(data_dict['phi'][0])
    print(len(data_dict['phi']))
    ######### todo - get dictionary of phi, f=1 or 0, t. Save as npy file
    ######### in mathematical side, I need to think of how to go from simulated f(t) to p(f,t) (sparse, noisy data)
    ############ --> the quality of p(f, t) is all bad unless nsim is very large. But I can stil pick some points (f_i, t_i)
    ## to make as good point estimate as possible based on the f(t)'s (e.g. near equilibria at relatively large t (t > 1/s))

