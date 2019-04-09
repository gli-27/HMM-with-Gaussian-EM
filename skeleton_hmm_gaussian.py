#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "../data" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    train_xs, dev_xs = parse_data(args)
    cov = np.cov(train_xs, rowvar=False)
    if args.cluster_num:
        mus = 10*np.random.standard_normal((args.cluster_num, 2))
        if not args.tied:
            sigmas = np.zeros([args.cluster_num, 2, 2])
            for k in range(0, args.cluster_num):
                    sigmas[k] = 3*cov + 10
        else:
            sigmas = 3*cov + 10
        transitions = np.zeros([args.cluster_num,args.cluster_num]) #transitions[i][j] = probability of moving from cluster i to cluster j
        for i in range(0, args.cluster_num):
            transitions[i] = np.random.dirichlet(np.ones(args.cluster_num), size=1)
        initials = np.random.dirichlet(np.ones(args.cluster_num), size=1).reshape(-1, 1) #probability for starting in each state
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        #raise NotImplementedError #remove when random initialization is implemented
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    
    class Model():
        def __init__(self):
            self.initials = initials
            self.mus = mus
            self.sigmas = sigmas
            self.initials = initials
            self.transitions = transitions
            self.alphas = None
            self.betas = None
            self.gamma = None
            self.xi = None
            self.likelihood = 0.0

        def estep(self, data, args):
            from scipy.stats import multivariate_normal
            gamma = np.zeros([len(data), args.cluster_num])
            xi = np.zeros([len(data), args.cluster_num, args.cluster_num])
            self.alphas, self.likelihood = forward(self, data, args)
            self.betas = backward(self, data, args)
            for k in range(args.cluster_num):
                gamma[:,k] = np.multiply(self.alphas[:,k], self.betas[:,k])/np.sum(np.multiply(self.alphas, self.betas), axis=1)
            for t in range(0, len(data)-1):
                for i in range(args.cluster_num):
                    for j in range(args.cluster_num):
                        if args.tied:
                            xi[t,i,j] = self.alphas[t,i]*self.transitions[i,j]*multivariate_normal(mean=self.mus[j], cov=self.sigmas).pdf(data[t+1])*self.betas[t+1,j]
                        else:
                            xi[t, i, j] = self.alphas[t, i] * self.transitions[i,j] * multivariate_normal(
                                mean=self.mus[j], cov=self.sigmas[j]).pdf(data[t+1]) * self.betas[t+1, j]
                xi[t,:,:] /= np.sum(xi[t,:,:])
            self.gamma = gamma
            self.xi = xi
            pass

        def mstep(self, data, args):
            self.initials = self.gamma[0,:]
            self.transitions = np.zeros([args.cluster_num, args.cluster_num])
            for t in range(len(data)):
                for i in range(args.cluster_num):
                    for j in range(args.cluster_num):
                        self.transitions[i,j] += self.xi[t,i,j]
                    self.mus[i] = np.dot(self.gamma[:,i].T, data)/np.sum(self.gamma[:,i])
                    if args.tied:
                        self.sigmas += np.dot(np.multiply(self.gamma[:, i].reshape(-1, 1), data[:] - self.mus[i]).T,
                                              (data[:] - self.mus[i])) / np.sum(self.gamma[:, i]) / args.cluster_num
                    else:
                        self.sigmas[i] = np.dot(np.multiply(self.gamma[:, i].reshape(-1, 1), data[:] - self.mus[i]).T,
                                                (data[:] - self.mus[i])) / np.sum(self.gamma[:, i])
            for k in range(args.cluster_num):
                self.transitions[:,k] /= np.sum(self.gamma, axis=0)

    model = Model()
    #raise NotImplementedError #remove when model initialization is implemented
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data),args.cluster_num))
    model.alphas = np.zeros((len(data),args.cluster_num))
    log_likelihood = 0.0
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.

    for k in range(args.cluster_num):
        if args.tied:
            alphas[0,k] = model.initials[k]*multivariate_normal(mean=model.mus[k], cov=model.sigmas).pdf(data[0,:])
        else:
            alphas[0,k] = model.initials[k]*multivariate_normal(mean=model.mus[k], cov=model.sigmas[k]).pdf(data[0,:])
    log_likelihood += log(np.sum(alphas[0,:]))
    alphas[0,:] /= np.sum(alphas[0,:])

    for t in range(1, len(data)):
        for i in range(args.cluster_num):
            for j in range(args.cluster_num):
                if args.tied:
                    alphas[t,i] += alphas[t-1,j]*model.transitions[j,i]*multivariate_normal(mean=model.mus[i], cov=model.sigmas).pdf(data[t])
                else:
                    alphas[t, i] += alphas[t-1,j]*model.transitions[j,i] * multivariate_normal(mean=model.mus[i], cov=model.sigmas[i]).pdf(data[t])
        log_likelihood += log(np.sum(alphas[t, :]))
        alphas[t,:] = alphas[t,:]/np.sum(alphas[t,:])

    #raise NotImplementedError
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))
    #TODO: Calculate and return backward probabilities (normalized like in forward before)

    for k in range(args.cluster_num):
        betas[len(data)-1, k] = 1
    betas[len(data)-1, :] /= np.sum(betas[len(data)-1, :])

    for t in range(len(data)-2, -1, -1):
        for i in range(args.cluster_num):
            for j in range(args.cluster_num):
                if args.tied:
                    betas[t, i] += model.transitions[i,j]*betas[t+1,j]*multivariate_normal(mean=model.mus[j], cov=model.sigmas).pdf(data[t+1])
                else:
                    betas[t, i] += model.transitions[i,j] * betas[t + 1, j] * multivariate_normal(mean=model.mus[j], cov=model.sigmas[j]).pdf(data[t+1])
        betas[t,:] /= np.sum(betas[t,:])


    #raise NotImplementedError
    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)

    while args.iterations:
        model.estep(train_xs, args)
        model.mstep(train_xs, args)
        args.iterations -= 1
    #raise NotImplementedError #remove when model training is implemented
    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    _, ll = forward(model, data, args)
    ll /= len(data)
    #raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    #initials = None
    #transitions = None
    #mus = None
    #sigmas = None
    initials = model.initials
    transitions = model.transitions
    mus = model.mus
    sigmas = model.sigmas
    #raise NotImplementedError #remove when parameter extraction is implemented
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')

    #parser.add_argument('--clusters_file', type=str, default='gaussian_hmm_smoketest_clusters.txt')
    #parser.add_argument('--cluster_num', type=int, default=2)

    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', default=True,help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
