HMM with Gaussian EM
--------------------
Guo Li 
Apr 9 2019 
--------------------


Usage
-----
The main implementation of HMM with Gaussian EM algorithm is in file skeleton_hmm_gaussian.py. Also, you can use smoke_test.py to test the implementation result of HMM_EM algorithm.


Briefly Description
-------------------
This project is an implementation of one application of hidden markov model, to learn the parameters to get the max log likelihood for current distribution. Learning parameters is to adjust the parameters of the hidden markov model given the oberserved sequence with EM algorithm (aka. Baum-Welch algorithm). There will be a distribution matrix of transition probability, which means the probability for one hidden state to another. Also, we will have an emission probability matrix, which is the probability of current state to the current obersevation variable. In this algorithm, we will have some intermediate variables such as alpha, beta, gamma and xi. Alphas and betas mean the forward and backward probability, which is similar as what we have learned in forward-backward algorithm in HMM. And gamma is the probability of being in one state i at time t for a specific sequence o. Xi is the the probability of being in one state i at time t and being in state j at time t + 1.


Extra args used in this assignment
----------------------------------
We actually only add one more parameter to the mutual group, --plot. When this parameter provided, the program will run and plot the analysis graph of iteration and cluster number's effect on thie problem. But note that this assignment we are not asked to deal with the singular matrix problem and according to experiments, providing this parameter and run the plot process will be likely to get stuck in singular matrix problem. Also, this will take some time to run and draw the graphs.


Method of Initialize Model
--------------------------
Almost the same with previous EM algorithm. Only add one transition matrix initilized by Dirichlet distribution.


Does the HMM model the data better than the original non-sequence model?
------------------------------------------------------------------------
Yes, from the analysis graph we can find out that, HMM model can obtain a relatively good result at the beginning of the iterations, even EM will get to a good result in just a few iteration times. Also, pure EM will easily stuck in some local optimal and cannot proceed increasing the precision. But for HMM model, it seems like we can obtain a better result (even just a few iteration times) with the increment iterations. 


What is the best number of states?
----------------------------------
From the analysis graph we plotted, it can be found that the number of states should be larger than 4. As the graphs tell us, when the state number is greater than 4, we will obtain result better than 3.8. And if the number larger enough, say 7, it will converge quickly. But this also depends on the situation. Generally speaking, the state number should be determined by experiments and if the number is too large, the dev likelihood will decrease as iterations, which is overfitting. 
