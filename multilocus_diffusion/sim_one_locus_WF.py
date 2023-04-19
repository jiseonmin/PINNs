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
    fseries = []
    tseries = []
    n = n0
    t = 0
    fseries.append(n/N)
    tseries.append(t)
    while n > 0 and n < N:
        n = np.random.binomial(N, n*(1+s) / (N-n + n*(1+s)))
        t += 1
        fseries.append(n/N)
        tseries.append(t)
    return fseries, tseries

if __name__ == '__main__':
    fendlist = []
    tendlist = []
    data = {'f': [], 't': []}
    n0 = int(args.f0 * args.N)
    nsim = 0
    while nsim < args.nsim:
        fseries, tseries = simulate_WF(args.N, n0, args.s)
        data['f'].extend(fseries)
        data['t'].extend(tseries)
        nsim += 1
        tendlist.append(tseries[-1])
        fendlist.append(fseries[-1])

    np.save("single_locus_FP_data.npy", data)
    print(np.max(tendlist))
    fmatrix = np.broadcast_to(fendlist, (np.max(tendlist) + 1, len(fendlist)))
    tmatrix = np.broadcast_to(range(np.max(tendlist) + 1), (len(fendlist), np.max(tendlist) + 1)).T
    print(fmatrix.shape)
    fmatrix.flags.writeable = True
    fmatrix2 = fmatrix.copy()
    t0array = np.argwhere(np.array(data['t']) == 0).flatten()
    print(t0array)
    print(np.array(data['f'])[t0array])
    for i in range(len(t0array) - 1):
        print(i)
        t0 = t0array[i]
        tend = t0array[i+1]
        fmatrix2[0:tend-t0, i] = np.array(data['f'][t0:tend])
    last_tend = len(data['f'])
    fmatrix2[0:last_tend-t0array[-1], -1] = np.array(data['f'][t0array[-1]:last_tend])
    np.save("single_locus_FP_fmatrix.npy", fmatrix2)

## A row in fmatrix represent a snapshot of allele frequency at i-th time point.