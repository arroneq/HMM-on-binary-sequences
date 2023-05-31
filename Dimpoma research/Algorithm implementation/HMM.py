from __future__ import annotations    # enable using function specifications
import numpy.typing as npt            # https://numpy.org/devdocs/reference/typing.html
import typing                         # https://docs.python.org/3/library/typing.html#typing.Union 

import numpy as np
from scipy.spatial import distance
import itertools
import random

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import copy

from numba import jit, cuda

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# Set of key computing functions

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

@jit(target_backend='cuda', forceobj=True)
def decimal(n: str):
    """ 
    Return int decimal number of n

    Parameters
    ----------
    n : string

    Examples
    --------
    >>> decimal("010")
    2
    """

    return int(n,2)

@jit(target_backend='cuda', forceobj=True)
def binary(n: int, N: int):
    """ 
    Return string N-character binary number of n

    Parameters
    ----------
    n : int
    N : int
        Dimention of any state vector  

    Examples
    --------
    >>> binary(2,3)
    "010"
    """
    
    bit = [(n>>k)&1 for k in range(0,N)]
    return ''.join([str(x) for x in bit])

def generate_hidden_chain(x0: str, p: float, N: int, T:int) -> list[str]:
    """ 
    Return chain of hidden states ("x_real")
    
    Parameters
    ----------
    x0: string
        First initial state
    p : float
        Transition probability of symbols ("0"->"0", "0"->"1", "1"->"0", "1"->"1")
    N : int
        Dimention of any state vector
    T : int
        Length of a chain 

    Returns
    -------
    x : string array(T,)
        Chain of hidden states
    """
    
    x = [x0]

    for t in range(T-1):
        # from "001" to [0,0,1] 
        state = [int(list(x[len(x)-1])[i]) for i in range(N)]

        random_index = random.choice([i for i in range(len(x[0]))])
        offer = copy.deepcopy(state[random_index])

        u = random.uniform(0,1)
        if u <= p:
            state[random_index] = offer
        else:
            state[random_index] = int(not(offer))

        # from [0,0,1] to "001"
        state = [str(state[i]) for i in range(N)]

        x.append(''.join(state))

    return x

def distort_hidden_chain(x: list[str], T: int, q: list[float], I: list[list[int]]):
    """ 
    Return distorted hidden state chain
    
    Parameters
    ----------
    x : string array(T,)
        Original chain of hidden states
    T : int
        Length of a chain 
    q : float array(len(I),)
        Distortion coefficients
    I : int array
        Set of observed indices

    Returns
    -------
    x_distorted : string array(T,)
                  Distorted hidden state chain
    """

    x_distorted = copy.deepcopy(x)

    for t in range(T):
        for j in range(len(I)):
            for i in I[j]:
                u = random.uniform(0,1)
                if u <= q[j]:
                    x_distorted[t] = list(x_distorted[t])
                    x_distorted[t][i] = str(1 - int(x_distorted[t][i]))
                    x_distorted[t] = ''.join(x_distorted[t])

    return x_distorted

def collect_observations(
        x: list[str], 
        I: list[list[int]], 
        T: int, enumerate: bool
    ) -> typing.Union[list[str], npt.NDArray[np.int64]]:

    """ 
    Return collected observations
    
    Parameters
    ----------
    x         : string array(T,)
                Chain of hidden states
    I         : int array
                Set of observed indices
    T         : int
                Length of a chain
    enumerate : bool
                Enumerate output observations array or not        
    
    Returns
    -------
    y : int array(T,)
        Enumerated chain of observations
    """

    y = []

    for t in range(T):
        y_t = []
        for j in range(len(I)):
            y_t.append(sum([int(list(x[t])[k]) for k in I[j]]))    
        y.append(y_t)

    if enumerate == False:
        return y
    
    if enumerate == True:
        """
        For now observations look like:
        >>> collect_observations(["011","010","000", "100"], [[0],[1,2]], 4)
        [[0,2],[0,1],[0,0],[1,0]]

        Let's enumerate items: y = [2,1,0,4]
        """

        args = [range(j+1) for j in [len(I[i]) for i in range(len(I))]] 
        # e.g. if len(I[0]) == 2, then possible values of observations are 0,1,2
        apo = [list(i) for i in itertools.product(*args)] # all_possible_оbservations
        
        for t in range(T):
            y[t] = apo.index(y[t])

        return np.asarray(y)

def flatten_observations(y: npt.NDArray[np.int64]):
    """ 
    Return string chain of observations

    Parameters
    ----------
    y : int array(T,)
        Enumerated chain of observations

    Returns
    -------
    y : str array(T,)
        String chain of observations
    
    Examples
    --------
    >>> flatten_observations([0,1],[1,1],[1,1],[1,0]])
    ["10","11","11","10"]
    """

    y_flattened = ["" for i in range(len(y))]

    for i in range(len(y)):
        for j in range(len(y[0])):
            y_flattened[i] += str(y[i][j])

    return y_flattened

@jit(target_backend='cuda', forceobj=True)
def probability_measures_of_HMM(
        p: float, 
        N: int, 
        I: list[list[int]], 
        *args: list[float]
    ) -> tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    
    """ 
    Return probability measures (pi,A,B) of a hidden Markov model

    Parameters
    ----------
    p     : float
            Transition probability of symbols ("0"->"0", "0"->"1", "1"->"0", "1"->"1"). This parameter is used for creating transition matrix A
    N     : int
            Dimention of any state vector
    I     : int array
            Set of observed indices
    *args : float array(len(I),)  
            Distortion coefficients q, mandatory only for distorted observations

    Returns
    -------
    pi : float array(pow(2,N),)
         Initial distribution
    A  : float array(pow(2,N),pow(2,N))
         Transition matrix
    B  : float array(pow(2,N), depends on I)
         Emission matrix
    """

    arguments = [range(j+1) for j in [len(I[i]) for i in range(len(I))]] 
    # e.g. if len(I[0]) == 2, then possible observated values are 0,1,2
    apo = [list(i) for i in itertools.product(*arguments)] # apo stands for all_possible_оbservations

    pi = np.array([1/pow(2,N) for i in range(pow(2,N))])

    A = np.zeros((pow(2,N),pow(2,N)))
    for i in range(len(A)):
        for j in range(len(A[0])):
            if distance.hamming(list(binary(i,N)),list(binary(j,N)))*N == 0:
                A[i][j] = p
            elif distance.hamming(list(binary(i,N)),list(binary(j,N)))*N == 1:
                A[i][j] = (1-p)/N

    if len(args) == 0:
        B = np.zeros((pow(2,N),len(apo)))
        for i in range(len(B)):
            tobo = [] # the_only_possible_оbservation
            for j in range(len(I)):
                tobo.append(sum([int(list(binary(i,N))[k]) for k in I[j]]))
            B[i][apo.index(tobo)] = 1.0

    if len(args) == 1:
        q = args[0]

        B = np.ones((pow(2,N),len(apo)))
        for i in range(len(B)):
            for j in range(len(B[0])):
                y = apo[j]

                for m in range(len(I)):
                    n11 = sum([int(list(binary(i,N))[u]) for u in I[m]])
                    n01 = len(I[m]) - n11

                    item = 0.0
                    for k in [u for u in range(max(0,y[m]-n11), min(y[m],n01)+1)]:
                        item += np.math.comb(n01,k)*pow(q[m],k)*pow(1-q[m],n01-k) * np.math.comb(n11,y[m]-k)*pow(1-q[m],y[m]-k)*pow(q[m],n11-y[m]+k)

                    B[i][j] *= item

    return pi,A,B

@jit(nopython=True)
def alpha_calculation(
        y: npt.NDArray[np.int64],
        m: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        T: int,
        *args: list[float]
    ):
    
    """
    Return forward algorithm coefficients

    Parameters
    ----------
    y     : int array(T,)
            Chain of observations
    m     : float array(pow(2,N),)
            Initial distribution
    A     : float array(pow(2,N),pow(2,N))
            Transition matrix
    B     : float array(pow(2,N), depends on I)
            Emission matrix
    T     : int
            Length of a chain   
    *args : coefficients of scaling (optional, used only for scaled forward algorithm)

    Returns
    -------
    alpha  : float array(pow(2,N),T)
             Forward algorithm coefficients
    P      : float
             Probability of P(Y=y)
    scaler : float array(pow(2,N),T)
             Coefficients of scaling (optional, used only for scaled forward algorithm)
    """

    alpha = np.zeros((T, len(B)))

    if len(args) == 0:
        for t in range(T):
            for i in range(len(B)):
                if t == 0: 
                    alpha[t][i] = m[i]*B[i][y[t]]
                else:
                    aA = 0.0
                    for j in range(len(B)):
                        aA += alpha[t-1][j]*A[j][i]
                    alpha[t][i] = aA*B[i][y[t]]

        P = 0
        for i in range(len(alpha[T-1])):
            P += alpha[T-1][i]

        return alpha, P

    if len(args) != 0:
        for t in range(T):
            for i in range(len(B)):
                if t == 0: 
                    alpha[t][i] = m[i]*B[i][y[t]]
                    args[0][t] += alpha[t][i]
                else:
                    aA = 0.0
                    for j in range(len(B)):
                        aA += alpha[t-1][j]*A[j][i]
                    alpha[t][i] = aA*B[i][y[t]]
                    args[0][t] += alpha[t][i]
            
            for i in range(len(B)):
                alpha[t][i] = alpha[t][i]/args[0][t]

        P = 0
        for t in range(T):
            P += np.log(args[0][t]) 

        return alpha, P, args[0]

@jit(nopython=True)
def beta_calculation(
        y: npt.NDArray[np.int64],
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        T: int,
        *args: list[float]
    ):
    
    """
    Return backward algorithm coefficients

    Parameters
    ----------
    y     : int array(T,)
            Chain of observations
    A     : float array(pow(2,N),pow(2,N))
            Transition matrix
    B     : float array(pow(2,N), depends on I)
            Emission matrix
    T     : int
            Length of a chain   
    *args : coefficients of scaling (optional, used only for scaled backward algorithm)

    Returns
    -------
    beta : float array(pow(2,N),T)
           Backward algorithm coefficients
    """

    beta = np.zeros((T, len(B)))

    if len(args) == 0:
        for t in range(T-1, -1, -1):
            for i in range(len(B)):
                if t == T-1: 
                    beta[t][i] = 1
                else:
                    bAB = 0.0
                    for j in range(len(B)):
                        bAB += beta[t+1][j]*A[i][j]*B[j][y[t+1]]
                    beta[t][i] = bAB

    if len(args) != 0:
        for t in range(T-1, -1, -1):
            for i in range(len(B)):
                if t == T-1: 
                    beta[t][i] = 1
                else:
                    bAB = 0.0
                    for j in range(len(B)):
                        bAB += beta[t+1][j]*A[i][j]*B[j][y[t+1]]
                    beta[t][i] = bAB

            for i in range(len(B)):
                beta[t][i] = beta[t][i]/args[0][t]

    return beta

@jit(nopython=True)
def calculate_p_numerator(
        y: npt.NDArray[np.int64],
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.int64],
        A: npt.NDArray[np.int64],
        B: npt.NDArray[np.int64],
        N: int,
        T: int
    ) -> float:
    
    """
    Return numerator of p estimation

    Parameters
    ----------
    y      : int array(T,)
             Chain of observations
    alpha  : float array(pow(2,N),T)
             Forward algorithm coefficients
    beta   : float array(pow(2,N),T)
             Backward algorithm coefficients
    A      : float array(pow(2,N),pow(2,N))
             Transition matrix
    B      : float array(pow(2,N), depends on I)
             Emission matrix
    N      : int
             Dimention of any state vector 
    T      : int
             Length of a chain   

    Returns
    -------
    output : float
    """

    exs = 0.0 # external_sum
    
    for t in range(T-1):
        ins = 0.0 # internal_sum
        for j in range(pow(2,N)):
            ins += alpha[t][j]*beta[t+1][j]*A[j][j]*B[j][y[t+1]]
        exs += ins

    return exs

@jit(nopython=True)
def calculate_p_denominator(alpha: npt.NDArray[np.float64], N: int, T: int) -> float:
    """
    Return denominator of p estimation

    Parameters
    ----------
    alpha  : float array(pow(2,N),T)
             Forward algorithm coefficients
    N      : int
             Dimention of any state vector 
    T      : int
             Length of a chain   

    Returns
    -------
    output : float
    """

    sum = 0.0
    for i in range(pow(2,N)):
        sum += alpha[T-1][i]

    return (T-1)*sum

def calculate_q_estimation_part_1(
        y: npt.NDArray[np.int64],
        qj: float,
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.int64],
        A: npt.NDArray[np.int64],
        N: int,
        T: int,
        I: list[list[int]],
        j: int
    ) -> float:
    
    """
    Return first part of q estimation

    Parameters
    ----------
    y        : int array(T,)
               Chain of observations
    qj       : float
               Current approximation of q[j] distortion coefficient
    alpha    : float array(pow(2,N),T)
               Forward algorithm coefficients
    beta     : float array(pow(2,N),T)
               Backward algorithm coefficients
    A        : float array(pow(2,N),pow(2,N))
               Transition matrix
    N        : int
               Dimention of any state vector 
    T        : int
               Length of a chain   
    I        : int array
               Set of observed indices
    j        : int
               Index of current q[j] distortion coefficient

    Returns
    -------
    output : float
    """

    def P_сonditional_probability_x_i_distorted(apo,xt,yt,i,j,qj,N,I):
        y = apo[yt]
        
        if y[j] + int(list(binary(xt,N))[i]) - 1 < 0 or y[j] + int(list(binary(xt,N))[i]) - 1 > len(I[j]) - 1:
            return 0.0
        
        else:
            element = 1.0
            for m in range(len(I)):
                if m != j:
                    n11 = sum([int(list(binary(xt,N))[u]) for u in I[m]])
                    n01 = len(I[m]) - n11

                    item = 0.0
                    for k in [u for u in range(max(0,y[m]-n11), min(y[m],n01)+1)]:
                        item += np.math.comb(n01,k)*pow(qj,k)*pow(1-qj,n01-k) * np.math.comb(n11,y[m]-k)*pow(1-qj,y[m]-k)*pow(qj,n11-y[m]+k)

                if m == j:
                    n11 = sum([int(list(binary(xt,N))[u]) for u in I[m] if u != i])
                    n01 = len(I[m]) - n11 - 1

                    item = 0.0
                    for k in [u for u in range(max(0,y[m]-1+int(list(binary(xt,N))[i])-n11), min(y[m]-1+int(list(binary(xt,N))[i]),n01)+1)]:
                        item += np.math.comb(n01,k)*pow(qj,k)*pow(1-qj,n01-k) * np.math.comb(n11,y[m]-1+int(list(binary(xt,N))[i])-k)*pow(1-qj,y[m]-1+int(list(binary(xt,N))[i])-k)*pow(qj,n11-y[m]+1-int(list(binary(xt,N))[i])+k)

                element *= item
            
            return element

    args = [range(v+1) for v in [len(I[u]) for u in range(len(I))]] 
    # e.g. if len(I[0]) == 2, then possible values of observations are 0,1,2
    apo = [list(u) for u in itertools.product(*args)] # all_possible_оbservations

    ins_t = 0.0 # internal_sum
    for t in range(T):
        ins_xt = 0.0 # one more internal sum
        for xt in range(pow(2,N)):
            ins_aA = 0.0 # another internal sum
            for xt_previous in range(pow(2,N)):
                ins_aA += alpha[t-1][xt_previous]*A[xt_previous][xt]
            
            ins_P = 0.0
            for i in I[j]:
                ins_P += P_сonditional_probability_x_i_distorted(apo,xt,y[t],i,j,qj,N,I)

            if t == 0:
                ins_xt += (1/pow(2,N))*qj*ins_P*beta[t][xt]
            else:
                ins_xt += ins_aA*qj*ins_P*beta[t][xt]
        ins_t += ins_xt
    
    output = copy.deepcopy(ins_t)

    return output

def calculate_q_estimation_part_2(
        y: npt.NDArray[np.int64],
        qj: float,
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.int64],
        A: npt.NDArray[np.int64],
        N: int,
        T: int,        
        I: list[list[int]],
        j: int
    ) -> float:
    
    """
    Return second part of q estimation

    Parameters
    ----------
    y        : int array(T,)
               Chain of observations
    qj       : float
               Current approximation of q[j] distortion coefficient
    alpha    : float array(pow(2,N),T)
               Forward algorithm coefficients
    beta     : float array(pow(2,N),T)
               Backward algorithm coefficients
    A        : float array(pow(2,N),pow(2,N))
               Transition matrix
    N        : int
               Dimention of any state vector 
    T        : int
               Length of a chain   
    I        : int array
               Set of observed indices
    j        : int
               Index of current q[j] distortion coefficient

    Returns
    -------
    output : float
    """
    def P_сonditional_probability_x_i_not_distorted(apo,xt,yt,i,j,qj,N,I):
        y = apo[yt]
        
        if y[j] - int(list(binary(xt,N))[i]) < 0 or y[j] - int(list(binary(xt,N))[i]) > len(I[j]) - 1:
            return 0.0
        
        else:
            element = 1.0
            for m in range(len(I)):
                if m != j:
                    n11 = sum([int(list(binary(xt,N))[u]) for u in I[m]])
                    n01 = len(I[m]) - n11

                    item = 0.0
                    for k in [u for u in range(max(0,y[m]-n11), min(y[m],n01)+1)]:
                        item += np.math.comb(n01,k)*pow(qj,k)*pow(1-qj,n01-k) * np.math.comb(n11,y[m]-k)*pow(1-qj,y[m]-k)*pow(qj,n11-y[m]+k)

                if m == j:
                    n11 = sum([int(list(binary(xt,N))[u]) for u in I[m] if u != i])
                    n01 = len(I[m]) - n11 - 1

                    item = 0.0
                    for k in [u for u in range(max(0,y[m]-int(list(binary(xt,N))[i])-n11), min(y[m]-int(list(binary(xt,N))[i]),n01)+1)]:
                        item += np.math.comb(n01,k)*pow(qj,k)*pow(1-qj,n01-k) * np.math.comb(n11,y[m]-int(list(binary(xt,N))[i])-k)*pow(1-qj,y[m]-int(list(binary(xt,N))[i])-k)*pow(qj,n11-y[m]+int(list(binary(xt,N))[i])+k)

                element *= item
            
            return element

    args = [range(v+1) for v in [len(I[u]) for u in range(len(I))]] 
    # e.g. if len(I[0]) == 2, then possible values of observations are 0,1,2
    apo = [list(u) for u in itertools.product(*args)] # all_possible_оbservations

    ins_t = 0.0 # internal_sum
    for t in range(T):
        ins_xt = 0.0 # one more internal sum
        for xt in range(pow(2,N)):
            ins_aA = 0.0 # another internal sum
            for xt_previous in range(pow(2,N)):
                ins_aA += alpha[t-1][xt_previous]*A[xt_previous][xt]
            
            ins_P = 0.0
            for i in I[j]:
                ins_P += P_сonditional_probability_x_i_not_distorted(apo,xt,y[t],i,j,qj,N,I)

            if t == 0:
                ins_xt += (1/pow(2,N))*(1-qj)*ins_P*beta[t][xt]
            else:
                ins_xt += ins_aA*(1-qj)*ins_P*beta[t][xt]
        ins_t += ins_xt
    
    output = copy.deepcopy(ins_t)

    return output

def learning_algorithm(
        y: npt.NDArray[np.int64],
        N: int,
        T: int,
        I: list[list[int]],
        estimator: typing.Literal[
            "parameter p estimation task (distortion-free model)", 
            "parameter p estimation task (model with distortion)",
            "parameter p and coefficients q estimation task"
        ],
        p0: float,
        q0: float,
        criterion: typing.Literal["N iterations", "increments"],
        scaling: bool,
        window: object
    ):
    
    """
    Return learning algorithm results (estimated parameters)

    Parameters
    ----------
    y         : int array(T,)
                Chain of observations
    N         : int
                Dimention of any state vector 
    T         : int
                Length of a chain
    I         : int array
                Set of observed indices
    estimator : string
                Type of estimation (only p or both p & q estimation)    
    p0        : float
                Initial approximation of parameter p 
    q0        : float array(len(I),)
                Initial approximation of distortion coefficients q 
    criterion : string
                Criterion to stop learning algorithm
    scaling   : string
                Either to scale forward and backward coefficients or not
    window    : GUI object  

    Returns
    -------
    parameter                      : float array(n+1,2)
                                     List of estimated parameters [p,q] for each iteration
    joint_probabilities            : float array(n,)
                                     List of P(Y=y) probabilities (concerns to forward algorithm) 
    joint_probabilities_increments : float array(n+1,)
                                     List of P(Y=y) increments through iterations (concerns to forward algorithm) 
    """

    joint_probabilities = []
    joint_probabilities_increments = []

    if estimator == "parameter p and coefficients q estimation task":
        parameter = []
        parameter.append([p0, q0])
    
        p = copy.deepcopy(parameter[0][0])
        q = copy.deepcopy(parameter[0][1])
    else:
        parameter = []
        parameter.append([p0])

        p = copy.deepcopy(parameter[0][0])

    if criterion == "increments":
        number_of_iterations = 0

        while (number_of_iterations < 20 or abs(1-joint_probabilities[-1]/joint_probabilities[-2]) > 0.0001):
            if estimator == "parameter p estimation task (distortion-free model)":
                m,A,B = probability_measures_of_HMM(p,N,I)
            elif estimator == "parameter p estimation task (model with distortion)":
                m,A,B = probability_measures_of_HMM(p,N,I,q0)
            elif estimator == "parameter p and coefficients q estimation task":
                m,A,B = probability_measures_of_HMM(p,N,I,q)

            if scaling == "false":
                alpha,P = alpha_calculation(y,m,A,B,T)
                beta = beta_calculation(y,A,B,T)
            else:
                scaler = np.zeros(T)
                alpha,P,scaler = alpha_calculation(y,m,A,B,T,scaler)
                beta = beta_calculation(y,A,B,T,scaler)
                
            joint_probabilities.append(copy.deepcopy(P))

            if number_of_iterations == 0:
                joint_probabilities_increments.append(copy.deepcopy(P))
            else:
                joint_probabilities_increments.append(abs(copy.deepcopy(P) - joint_probabilities_increments[-1]))

            p_numerator = calculate_p_numerator(y,alpha,beta,A,B,N,T)
            p_denominator = calculate_p_denominator(alpha,N,T)
            p = p_numerator/p_denominator

            if np.isnan(p):
                print(f"Error p: k = {number_of_iterations+1}, [p0,q0] = {parameter[0]}")
                parameter = [parameter[i] for i in range(number_of_iterations+1)]
                joint_probabilities = [joint_probabilities[i] for i in range(number_of_iterations)]
                joint_probabilities_increments = [joint_probabilities_increments[i] for i in range(number_of_iterations+1)]
                break 

            parameter.append([copy.deepcopy(p)])

            if estimator == "parameter p and coefficients q estimation task":
                for j in range(len(I)):
                    part_1 = calculate_q_estimation_part_1(y,q[j],alpha,beta,A,N,T,I,j)
                    part_2 = calculate_q_estimation_part_2(y,q[j],alpha,beta,A,N,T,I,j)

                    qj_numerator = copy.deepcopy(part_1)
                    qj_denominator = copy.deepcopy(part_1) + copy.deepcopy(part_2)

                    q[j] = qj_numerator/qj_denominator

                    if np.isnan(q[j]):
                        print(f"Error q: k = {number_of_iterations+1}, [p0,q0] = {parameter[0]}")
                        parameter = [parameter[i] for i in range(number_of_iterations+1)]
                        joint_probabilities = [joint_probabilities[i] for i in range(number_of_iterations)]
                        joint_probabilities_increments = [joint_probabilities_increments[i] for i in range(number_of_iterations+1)]
                        break 

                parameter[number_of_iterations + 1].append(copy.deepcopy(q))

            number_of_iterations += 1

            window["-PROGRESS BAR PERCENT-"].update(f"i : {number_of_iterations}")
            window["-PROGRESS BAR-"].update(0)
            window.refresh()

            if number_of_iterations >= 150:
                break
    else:
        number_of_iterations = int(criterion[0:2])

        for k in range(number_of_iterations):
            if estimator == "parameter p estimation task (distortion-free model)":
                m,A,B = probability_measures_of_HMM(p,N,I)
            elif estimator == "parameter p estimation task (model with distortion)":
                m,A,B = probability_measures_of_HMM(p,N,I,q0)
            elif estimator == "parameter p and coefficients q estimation task":
                m,A,B = probability_measures_of_HMM(p,N,I,q)

            if scaling == "false":
                alpha,P = alpha_calculation(y,m,A,B,T)
                beta = beta_calculation(y,A,B,T)
            else:
                # scaler = np.zeros(T, dtype=np.float128)
                scaler = np.zeros(T)
                alpha,P,scaler = alpha_calculation(y,m,A,B,T,scaler)
                beta = beta_calculation(y,A,B,T,scaler)
                
            joint_probabilities.append(P)

            if k == 0:
                joint_probabilities_increments.append(copy.deepcopy(P))
            else:
                joint_probabilities_increments.append(abs(copy.deepcopy(P) - joint_probabilities_increments[-1]))

            p_numerator = calculate_p_numerator(y,alpha,beta,A,B,N,T)
            p_denominator = calculate_p_denominator(alpha,N,T)
            p = p_numerator/p_denominator

            if np.isnan(p):
                parameter = [parameter[i] for i in range(k+1)]
                joint_probabilities = [joint_probabilities[i] for i in range(k)]
                joint_probabilities_increments = [joint_probabilities_increments[i] for i in range(k+1)]
                print(f"Error p: k = {k+1}")
                break 

            parameter.append([copy.deepcopy(p)])

            if estimator == "parameter p and coefficients q estimation task":
                for j in range(len(I)):
                    part_1 = calculate_q_estimation_part_1(y,q[j],alpha,beta,A,N,T,I,j)
                    part_2 = calculate_q_estimation_part_2(y,q[j],alpha,beta,A,N,T,I,j)

                    qj_numerator = copy.deepcopy(part_1)
                    qj_denominator = copy.deepcopy(part_1) + copy.deepcopy(part_2)

                    q[j] = qj_numerator/qj_denominator

                    if np.isnan(q[j]):
                        parameter = [parameter[i] for i in range(k+1)]
                        joint_probabilities = [joint_probabilities[i] for i in range(k)]
                        joint_probabilities_increments = [joint_probabilities_increments[i] for i in range(k+1)]
                        print(f"Error q: k = {k+1}")
                        break 

                parameter[k+1].append(copy.deepcopy(q))

            window["-PROGRESS BAR PERCENT-"].update(f"{((k+1)*100)//number_of_iterations}%")
            window["-PROGRESS BAR-"].update(((k+1)*100)//number_of_iterations)
            window.refresh()

    # joint_probabilities P(Y=y) for the last estimated parameters
    if estimator == "parameter p estimation task (distortion-free model)":
        m,A,B = probability_measures_of_HMM(p,N,I)
    elif estimator == "parameter p estimation task (model with distortion)":
        m,A,B = probability_measures_of_HMM(p,N,I,q0)
    elif estimator == "parameter p and coefficients q estimation task":
        m,A,B = probability_measures_of_HMM(p,N,I,q)

    if scaling == "false":
        alpha,P = alpha_calculation(y,m,A,B,T)
        beta = beta_calculation(y,A,B,T)
    else:
        scaler = np.zeros(T)
        alpha,P,scaler = alpha_calculation(y,m,A,B,T,scaler)
        beta = beta_calculation(y,A,B,T,scaler)
        
    joint_probabilities.append(copy.deepcopy(P))
    joint_probabilities_increments.append(abs(copy.deepcopy(P) - joint_probabilities_increments[-1]))        

    return parameter, joint_probabilities, joint_probabilities_increments

def statistical_p_estimation(
        y: npt.NDArray[np.int64],
        N: int,
        T: int,
        I: list[list[int]],
        estimator: typing.Literal[
            "parameter p estimation task (distortion-free model)", 
            "parameter p estimation task (model with distortion)",
            "parameter p and coefficients q estimation task"
        ]
    ) -> float:
    
    """
    Return statistical estimation of parameter p

    Parameters
    ----------
    y         : int array(T,)
                Chain of observations
    N         : int
                Dimention of any state vector 
    T         : int
                Length of a chain
    I         : int array
                Set of observed indices
    estimator : string
                Type of estimation (only p or both p & q estimation) 

    Returns
    -------
    parameter : float
                Estimated parameter p (statistical estimation)
    """

    if estimator == "parameter p estimation task (distortion-free model)":
        counter = 0
        for t in range(T-1):
            if y[t] == y[t+1]:
                counter += 1

        I_intersection = len(set(sum(I,[])))
        parameter = 1 - (N/I_intersection) * (1 - counter/(T-1))

        return parameter

    else:
        return "none"

@jit(target_backend='cuda', forceobj=True)
def viterbi(
        y: npt.NDArray[np.int64],
        m: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        T: int
    ) -> npt.NDArray[np.int64]:

    """
    Return decoded state chain

    Parameters
    ----------
    y : int array(T,)
        Chain of observations
    m : float array(pow(2,N),)
        Initial distribution
    A : float array(pow(2,N),pow(2,N))
        Transition matrix
    B : float array(pow(2,N), depends on I)
        Emission matrix
    T : int
        Length of a chain   
    *args : coefficients of scaling (optional, used only for scaled forward algorithm)

    Returns
    -------
    x : int array(T,)
        Decoded enumarated state chain
    """

    delta = [[0.0 for i in range(len(B))] for t in range(T)]
    psi = [[0 for i in range(len(B))] for t in range(T)]

    for t in range(T):
        for i in range(len(B)):
            if t == 0: 
                delta[t][i] = m[i]*B[i][y[t]]
            else:
                dA = []
                for j in range(len(B)):
                    dA.append(delta[t-1][j]*A[j][i]*B[i][y[t]])
                delta[t][i] = max(dA)
                psi[t][i] = np.argmax(dA)

    x = []
    delta_hat = max(delta[T-1])
    x.append(np.argmax(delta[T-1]))

    for t in range(T-2, -1, -1):
        x.insert(0, psi[t+1][x[0]])

    return x

@jit(target_backend='cuda', forceobj=True)
def viterbi_algorithm(
        x_real: list[str],
        y: npt.NDArray[np.int64],
        N: int,
        T: int,
        I: list[list[int]],
        estimated_parameters: list[list[float]],
        estimator: typing.Literal[
            "parameter p estimation task (distortion-free model)", 
            "parameter p estimation task (model with distortion)",
            "parameter p and coefficients q estimation task"
        ],
        values: object
    ):
    
    """
    Return viterbi algorithm results: predicted hidden chain and several measures of accuracy 

    Parameters
    ----------
    x_real               : string array(T,)
                           Real chain of the hidden states
    y                    : int array(T,)
                           Chain of observations
    N                    : int
                           Dimention of any state vector 
    T                    : int
                           Length of a chain
    I                    : int array
                           Set of observed indices
    estimated_parameters : float array
                           Estimated parameters p (and q)
    estimator            : string
                           Type of estimation (only p or both p & q estimation)    
    values               : GUI object  

    Returns
    -------
    x_predicted                             : int array(T,)
                                              Decoded enumarated state chain
    x_hamming_distances, x_mismatch_indices : Arrays that keep information about a match level 
                                              between x_real and x_predicted   
    y_hamming_distances, y_mismatch_indices : Arrays that keep information about a match level 
                                              between y_real and y_predicted (based on x_predicted)   
    """
    
    # ------------------------ initialization ------------------------

    if estimator == "parameter p estimation task (distortion-free model)":
        m,A,B = probability_measures_of_HMM(estimated_parameters[0],N,I)
    elif estimator == "parameter p estimation task (model with distortion)":
        m,A,B = probability_measures_of_HMM(estimated_parameters[0],N,I,eval(values["-DISTORTION COEFFICIENTS-"]))
    elif estimator == "parameter p and coefficients q estimation task":
        m,A,B = probability_measures_of_HMM(estimated_parameters[0],N,I,estimated_parameters[1])

    # ---------------------- decoding algorithm ----------------------

    x_predicted = viterbi(y,m,A,B,T)
    x_predicted = [binary(x_predicted[i],N) for i in range(len(x_predicted))]

    # -------------- define errors between real and predicted states --------------

    x_hamming_distances = [0 for i in range(N+1)]
    x_mismatch_indices = [[] for i in range(N+1)]
    for t in range(T):
        for i in range(N+1): 
            if distance.hamming(list(x_real[t]),list(x_predicted[t]))*N == i:
                x_hamming_distances[i] += 1/T
                if i != 0:
                    x_mismatch_indices[i].append(define_mismatch_indices(x_real[t],x_predicted[t]))
    
    # hamming_distances_k[2] = [0.75, 0.25, 0, 0] --> hamming_distances_k[2] = [75,25,0,0]
    x_hamming_distances = [x_hamming_distances[i]*100 for i in range(N+1)]

    # mismatch_indices_k[2] = [[1,2],[3,4],[1,4]] --> mismatch_indices_k[2] = [1,2,3,4,1,4]
    x_mismatch_indices = [sum(x_mismatch_indices[i],[]) for i in range(N+1)]

    # -------------- define errors between real and predicted observations --------------

    y_real = collect_observations(x_real,I,T,enumerate=False)
    y_predicted = collect_observations(x_predicted,I,T,enumerate=False)

    y_real = flatten_observations(y_real)
    y_predicted = flatten_observations(y_predicted)

    y_hamming_distances = [0 for i in range(len(I)+1)]
    y_mismatch_indices = [[] for i in range(len(I)+1)]
    for t in range(T):
        for i in range(len(I)+1): 
            if distance.hamming(list(y_real[t]),list(y_predicted[t]))*len(I) == i:
                y_hamming_distances[i] += 1/T
                if i != 0:
                    y_mismatch_indices[i].append(define_mismatch_indices(y_real[t],y_predicted[t]))

    # hamming_distances_k[2] = [0.75, 0.25, 0, 0] --> hamming_distances_k[2] = [75,25,0,0]
    y_hamming_distances = [y_hamming_distances[i]*100 for i in range(len(I)+1)]

    # mismatch_indices_k[2] = [[1,2],[3,4],[1,4]] --> mismatch_indices_k[2] = [1,2,3,4,1,4]
    y_mismatch_indices = [sum(y_mismatch_indices[i],[]) for i in range(len(I)+1)]

    return x_predicted, x_hamming_distances, x_mismatch_indices, y_hamming_distances, y_mismatch_indices

def estimate_implicit_indices(
        x_real: list[str],
        x_predicted: list[int],
        real_implicit_indices: list[int],
        estimate_length: typing.Literal["maximum", "сonsistent"],
        T0: int,
        T: int,
        N: int,
        metrics_type: typing.Literal["square", "weighted Jaccard"]
    ) -> tuple[int]:
    
    """
    Return estimation of implicit indices in a hidden chain 

    Parameters
    ----------
    x_real                     : string array(T,)
                                 Real chain of the hidden states
    x_predicted                : int array(T,)
                                 Decoded enumarated state chain
    real_implicit_indices      : int array
                                 Real set of implicit indices
    estimate_length            : string array()
                                 A way to estimate length of set of implicit indices: eigher by ["maximum"] method or by using ["сonsistent", p*] estimation
    T0                         : int
                                 Start estimation algorithm beginning with a T0 state (it means T0 states are considered to be for "overclocking")  
    T                          : int
                                 Length of a chain
    N                          : int
                                 Dimention of any state vector 
    metrics_type               : string
                                 Either "square" or "weighted Jaccard"

    Returns
    -------
    predicted_implicit_indices : int array(T,)
                                 Set of predicted_implicit_indices (considering an appropriate list of "0", "1", "2" etc. for given combination) 
    """

    phi_real = np.zeros(T, dtype=int)
    for t in range(T0,T):
        phi_real[t] = sum([int(list(x_real[t])[i]) for i in real_implicit_indices])   

    if estimate_length[0] == "maximum":
        estimated_length = max(phi_real)
    elif estimate_length[0] == "consistent":
        p = estimate_length[1]
        estimated_length = int((N/(1-p))*(1 - sum([1 for t in range(T0,T-1) if phi_real[t] == phi_real[t+1]])/(T-1)))

    offered_implicit_indices = list(itertools.combinations([i for i in range(N)], estimated_length))

    metric = np.zeros(len(offered_implicit_indices))
    for k in range(len(offered_implicit_indices)):
        phi_offered = np.zeros(T, dtype=int)
        for t in range(T0,T):
            phi_offered[t] = sum([int(list(x_predicted[t])[i]) for i in offered_implicit_indices[k]])
        
        metric[k] = define_distance(phi_real,phi_offered,metrics_type)

    min_metric_value = min(metric)
    argmin_metric_value = [index for index in range(len(metric)) if metric[index] == min_metric_value]

    predicted_implicit_indices = []
    for index in argmin_metric_value:
        predicted_implicit_indices.append(offered_implicit_indices[index])

    return predicted_implicit_indices[0] # collisions happen fairly rarely (only if T < 15)

def define_mismatch_indices(
        real: list[str],
        predicted: list[str]
    ) -> list[int]:
    
    """
    Return mismatch indices between original states/observations chain and the decoded one

    Parameters
    ----------
    real      : string array(T,)
                Original hidden states / observations chain
    predicted : string array(T,)
                Predicted hidden states / observations chain

    Returns
    -------
    indices : int array
              Mismatch indices between original and predicted chains
    """

    indices = []
    
    for i in range(len(real)):
        if real[i] != predicted[i]:
            indices.append(i)

    return indices

def define_groups_of_crossed_states(I: list[list[int]], N: int) -> list[int]:
    """ 
    Return intersected indices according to I 

    Parameters
    ----------
    I : int array
        Set of observed indices
    N : int
        Dimention of any state vector

    Returns
    -------
    cs : int array
         Intersected indices according to I
    """

    cs = [] # crossed_states
    os = sum([I[i] for i in range(len(I))],[]) # observed_states
    aps = [i for i in range(N)] # all_possible_states

    max_crossing_index = max([os.count(i) for i in aps])

    for j in range(0, max_crossing_index + 1):
        cs.append(list(set([os[i] for i in range(len(os)) if os.count(os[i]) == j])))

    ss = [aps[i] for i in range(len(aps)) if aps[i] not in os] # separated_states
    cs[0] = ss

    return cs

def define_distance(
        A: npt.NDArray[np.float64], 
        B: npt.NDArray[np.float64], 
        type: typing.Literal["square", "weighted Jaccard", "Wasserstein"],
        *args: float
    ):

    """ 
    Return a chosen distance between set A and set B
    """

    if type == "square":
        return sum([pow(a-b,2) for a,b in zip(A,B)])
    if type == "weighted Jaccard":
        return 1 - sum([min(a,b) for a,b in zip(A,B)])/sum([max(a,b) for a,b in zip(A,B)])
    if type == "Wasserstein":
        A = np.sort(A)
        B = np.sort(B)
        return pow(sum([pow(abs(a-b),args[0]) for a,b in zip(A,B)])/len(A),1/args[0])

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# Set of visualisation functions (no function specifications)

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def display_state(N):
    """ 
    Save figure of a state graph

    Parameters
    ----------
    N : int
        Dimention of any state vector
    """

    font = {
        "family": "serif",
        "size": 12,
    }

    left, bottom, width, height = 0.05, 0.05, 0.9, 0.9
    plt.figure(figsize=(6.4,4.8)).add_axes([left, bottom, width, height])

    # find the best points_in_line parameter for a nice looking graph (rectangle shape)
    for i in range(1, int(np.sqrt(N))+2):
        if N <= i**2: 
            pil = i
            break

    # convert states into points
    x = np.array([i%pil for i in range(N)])
    y = sorted(np.array([i//pil for i in range(N)])*(-1), reverse=True)

    plt.plot(x, y, marker = 'o', ms = 25, mec = 'black', mew=2, mfc = 'white', linestyle='None', color = 'black')

    # annotate points of graph
    i = 0
    for a,b in zip(x,y):
        index = "{" + f"{str(i)}" + "}"
        label = f"$x_{index}$"

        plt.annotate(label,             # this is the text
            (a,b),                      # these are the coordinates to position the label
            textcoords="offset points", # how to position the text
            xytext=(1,-1),              # distance from text to points (x,y)
            ha='center',                # horizontal alignment can be left, right or center
            va="center",                # vertical alignment can be bottom, top or center
            font = font,
        )
        i += 1

    plt.axis("off")
    plt.axis("equal")

    plt.savefig("Images/just_states.png")
    plt.close()

def display_graph(I,implicit_indices,N):
    """ 
    Save figure of a state graph according to I

    Parameters
    ----------
    I                : int array
                       Set of observed indices
    implicit_indices : int array
                       Set of implicit indices
    N                : int
                       Dimention of any state vector
    """

    font = {
        "family": "serif",
        "size": 12,
    }

    left, bottom, width, height = 0.05, 0.05, 0.9, 0.9
    plt.figure(figsize=(6.4,4.8)).add_axes([left, bottom, width, height])

    # find the best points_in_line parameter for a nice looking graph (rectangle shape)
    for i in range(1, int(np.sqrt(N))+2):
        if N <= i**2: 
            pil = i
            break

    # convert states into points
    u = np.array([i%pil for i in range(N)])
    v = sorted(np.array([i//pil for i in range(N)])*(-1), reverse=True)
    
    # add the first point in each set in order to make cycles
    for k in range(len(I)):
        x = [u[i] for i in I[k]]
        x.append(u[I[k][0]])

        y = [v[i] for i in I[k]]
        y.append(v[I[k][0]])

        plt.plot(x, y, 
                 marker = 'o', ms = 25, mec = 'black', mew=2, mfc = 'white', color = 'black')

    # find the cross states to emphasize them in a graph
    os = sum([I[i] for i in range(len(I))],[]) # observed_states
    cs = list(set([os[i] for i in range(len(os)) if os.count(os[i]) > 1])) # cross_states
    plt.plot([u[i] for i in cs], [v[i] for i in cs], 
              marker = 'o', ms = 25, mec = 'black', mew=5, mfc = 'white', 
              linestyle='None', color = 'black')

    # find separate states to emphasize them in a graph
    os = sum([I[i] for i in range(len(I))],[]) # observed_states
    aps = [i for i in range(N)] # all_possible_states
    ss = [aps[i] for i in range(len(aps)) if aps[i] not in os] # separated_states
    plt.plot([u[i] for i in ss], [v[i] for i in ss], 
             marker = 'o', ms = 25, mec = 'black', mew=2, mfc = 'white', 
             linestyle='None', color = 'black')

    # emphasize implicit indices if there are some
    if implicit_indices != "none":
        for i in eval(implicit_indices):
            if i in cs:
                plt.plot([u[i]], [v[i]], 
                    marker = 'o', ms = 25, mec = 'black', 
                    mew=5, mfc = 'lightgray', linestyle='None', color = 'black')
            else:
                plt.plot([u[i]], [v[i]], 
                    marker = 'o', ms = 25, mec = 'black', 
                    mew=2, mfc = 'lightgray', linestyle='None', color = 'black')

    # annotate points of graph
    i = 0
    for x,y in zip(u,v):
        index = "{" + f"{str(i)}" + "}"
        label = f"$x_{index}$"

        plt.annotate(label,
            (x,y),
            textcoords="offset points", 
            xytext=(1,-1), 
            ha='center', 
            va="center",
            color="black",
            font = font,
        )
        i += 1

    plt.axis("off")
    plt.axis("equal")
    
    plt.savefig("Images/graph.png")
    plt.close()

def display_convergence(parameters,joint_probabilities,p_statistical_estimation):
    """ 
    Show & save figure of convergence

    Parameters
    -------
    parameters               : float array(n,2)
                               List of estimated parameters [p,q] for each iteration
    joint_probabilities      : float array(n,)
                               List of P(Y=y) probabilities (concerns to forward algorithm) 
    p_statistical_estimation : float
                               Estimated parameter p, estimated by non-baum-welch algorithm
    """

    font = {
        "family": "serif",
        "size": 16
    }

    if len(parameters[-1]) == 1:
        plt.figure(figsize=(11,9)).suptitle("Збіжність алгоритму навчання протягом одного рестарту", fontsize=16, fontfamily="serif")

        plt.subplot(2,1,1)
        if len(joint_probabilities) <= 100:
            plt.plot([i for i in range(len(joint_probabilities))], joint_probabilities, "o-")
        else:
            plt.plot([i for i in range(len(joint_probabilities))], joint_probabilities)
        if len(joint_probabilities) % 10 == 0:
            plt.xticks([i*((len(joint_probabilities))//10) for i in range(11)])
        plt.xlabel(r"а) Значення ймовірностей $P_\lambda(Y=y)$ від ітерації до ітерації", y=1.05, loc="left", fontfamily="serif", fontsize=16)
        plt.grid(True, linestyle='-.')

        plt.subplot(2,1,2)
        if len(parameters) <= 100:
            plt.plot([i for i in range(len(parameters))], [parameters[i][0] for i in range(len(parameters))], "o-")
        else:
            plt.plot([i for i in range(len(parameters))], [parameters[i][0] for i in range(len(parameters))])
        plt.title(" ", y=1.05, font=font, loc="left")
        if len(parameters) % 10 == 0:
            plt.xticks([i*(len(parameters)//10) for i in range(11)])
        plt.xlabel(
                r"б) Значення параметра $p$ від ітерації до ітерації" + 
                "\n\n" + 
                f"Оцінка алгоритму навчання: p*={round(parameters[-1][0],8)}, статистична оцінка: p={round(p_statistical_estimation,8)}", 
                y=1.05, loc="left", fontfamily="serif", fontsize=16
            )
        plt.grid(True, linestyle='-.')

    if len(parameters[-1]) == 2 and len(parameters[-1][1]) == 2:
        plt.figure(figsize=(9.5,10)).suptitle("Збіжність алгоритму навчання протягом одного рестарту", fontsize=16, fontfamily="serif")

        plt.subplot(3,1,1)
        if len(joint_probabilities) <= 100:
            plt.plot([i for i in range(len(joint_probabilities))], joint_probabilities, "o-")
            plt.plot([0], joint_probabilities[0], "o-", color="red")
        else:
            plt.plot([i for i in range(len(joint_probabilities))], joint_probabilities)
            plt.plot([0], joint_probabilities[0], "o-", color="red")
        if len(joint_probabilities) % 10 == 0:
            plt.xticks([i*((len(joint_probabilities))//10) for i in range(11)])
        plt.xlabel(r"а) Значення ймовірностей $P_\lambda(Y=y)$ від ітерації до ітерації", y=1.05, loc="left", fontfamily="serif", fontsize=16)
        plt.grid(True, linestyle='-.')

        plt.subplot(3,1,2)
        if len(parameters) <= 100:
            plt.plot([i for i in range(len(parameters))], [parameters[i][0] for i in range(len(parameters))], "o-")
            plt.plot([0], [parameters[0][0]], "o", color="red")

        else:
            plt.plot([i for i in range(len(parameters))], [parameters[i][0] for i in range(len(parameters))])
            plt.plot([0], [parameters[0][0]], "o", color="red")
        plt.title(" ", y=1.05, font=font, loc="left")
        if len(parameters) % 10 == 0:
            plt.xticks([i*(len(parameters)//10) for i in range(11)])
        plt.xlabel(r"б) Значення параметра $p$ від ітерації до ітерації", y=1.05, loc="left", fontfamily="serif", fontsize=16)
        plt.grid(True, linestyle='-.')
        
        plt.subplot(3,1,3)
        if len(parameters) <= 100:
            plt.plot(
                [parameters[i][1][0] for i in range(len(parameters))], 
                [parameters[i][1][1] for i in range(len(parameters))], 
                "o-"
            )
            plt.plot([parameters[0][1][0]], [parameters[0][1][1]], "o-", color="red") 
        else:
            plt.plot(
                [parameters[i][1][0] for i in range(len(parameters))], 
                [parameters[i][1][1] for i in range(len(parameters))]
            )   
            plt.plot([parameters[0][1][0]], [parameters[0][1][1]], "o-", color="red")      
        plt.title(" ", y=1.05, font=font, loc="left")
        plt.xlabel(
            r"в) Значення параметра $q=(q_1,q_2)$ від ітерації до ітерації" + 
            "\n\n" + 
            f"Оцінка параметра $p*={round(parameters[-1][0],4)}$, " + 
            f"оцінка параметра $q*={[round(parameters[-1][1][i],4) for i in range(len(parameters[-1][1]))]}$", 
            y=1.05, loc="left", fontfamily="serif", fontsize=16
        )
        plt.grid(True, linestyle='-.')
    
    elif len(parameters[-1]) == 2 and len(parameters[-1][1]) != 2:
        plt.figure(figsize=(11,9)).suptitle("Збіжність алгоритму навчання протягом одного рестарту", fontsize=16, fontfamily="serif")

        plt.subplot(2,1,1)
        if len(joint_probabilities) <= 100:
            plt.plot([i for i in range(len(joint_probabilities))], joint_probabilities, "o-")
        else:
            plt.plot([i for i in range(len(joint_probabilities))], joint_probabilities)
        if len(joint_probabilities) % 10 == 0:
            plt.xticks([i*((len(joint_probabilities))//10) for i in range(11)])
        plt.xlabel(r"а) Значення ймовірностей $P_\lambda(Y=y)$ від ітерації до ітерації", y=1.05, loc="left", fontfamily="serif", fontsize=16)
        plt.grid(True, linestyle='-.')

        plt.subplot(2,1,2)
        if len(parameters) <= 100:
            plt.plot([i for i in range(len(parameters))], [parameters[i][0] for i in range(len(parameters))], "o-")
        else:
            plt.plot([i for i in range(len(parameters))], [parameters[i][0] for i in range(len(parameters))])
        plt.title(" ", y=1.05, font=font, loc="left")
        # plt.legend([r"значення параметра $p$"], prop = font, loc = "upper right")
        if len(parameters) % 10 == 0:
            plt.xticks([i*(len(parameters)//10) for i in range(11)])
        plt.xlabel(
            r"б) Значення параметра $p$ від ітерації до ітерації" + 
            "\n\n" + 
            f"Оцінка параметра $p*={round(parameters[-1][0],4)}$, " + 
            f"оцінка параметра $q*={[round(parameters[-1][1][i],4) for i in range(len(parameters[-1][1]))]}$", 
            y=1.05, loc="left", fontfamily="serif", fontsize=16
        )
        plt.grid(True, linestyle='-.')

    plt.tight_layout()
    plt.show()

    plt.savefig("Images/convergence.png")
    plt.close()

def display_estimated_parameters(estimated_parameters,MLE,p_statistical_estimation):
    """ 
    Show & save two plots:
        1) estimated probabilities p for each restart
        2) histogram of estimated probabilities p

    Parameters
    -------
    estimated_parameters     : float array(r,2)
                               List of estimated parameters [p,q] for each restart
    MLE                      : float array(r,)
                               List of P(Y=y) for each estimated probability (MLE -- maximum likelihood estimation)
    p_statistical_estimation : float array(r,)
                               List of estimated parameter p, estimated by non-baum-welch algorithm
    """

    font = {
        "family": "serif",
        "size": 15
    }

    if len(estimated_parameters[-1]) == 1:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13,5), 
            gridspec_kw={
                'width_ratios': [2.5, 1], 
                'bottom': 0.3,
                'left': 0.05,
                'right': 0.95,  
                'hspace': 0.005
            }
        )

        estimated_probabilities = [estimated_parameters[i][0] for i in range(len(estimated_parameters))]

        ax1.plot(
            [i+1 for i in range(len(estimated_probabilities))], 
            estimated_probabilities,
            marker="o",
            linestyle="-"
        )

        caption = r"Вибіркове середнє $\overline{p}=$" + f"{round(np.mean(estimated_probabilities),4)}, " + \
                  r"вибіркова дисперсія $S^2=$" + f"{round(np.var(estimated_probabilities),4)} | " + r"Cтатистична оцінка $\overline{p}=$" + f"{round(np.mean(p_statistical_estimation),4)}, " + r"$S^2=$" + f"{round(np.var(p_statistical_estimation),4)}" + "\n" + r"Оцінка, що максимізує $P(Y=y)$: $\widehat{p}=$" + f"{round(estimated_probabilities[np.argmax(MLE)],4)}"


        ax1.set_title(r"Значення параметрів $p^*$ від рестарту до рестарту", y=1.01, font=font)
        ax1.grid(True, linestyle='-.')
        ax1.text(
            0.01, -0.3, caption, 
            horizontalalignment='left',
            verticalalignment='center', 
            transform=ax1.transAxes,
            font=font
        )

        ax2.hist(
            estimated_probabilities, 
            weights=(1/len(estimated_probabilities))*np.ones_like(estimated_probabilities)
        )
        ax2.set_title(r"Гістограма значень $p^*$", y=1.01, font=font)
        ax2.grid(True, axis="y", linestyle='-.')

    if len(estimated_parameters[-1]) == 2 and len(estimated_parameters[-1][1]) == 2:
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(16,5), 
            gridspec_kw={
                'width_ratios': [2.5, 1, 1], 
                'bottom': 0.3,
                'left': 0.05,
                'right': 0.95,  
                'hspace': 0.005
            }
        )

        estimated_probabilities = [estimated_parameters[i][0] for i in range(len(estimated_parameters))]
        estimated_distortion_coefficients = [estimated_parameters[i][1] for i in range(len(estimated_parameters))]

        ax1.plot(
            [i+1 for i in range(len(estimated_probabilities))], 
            estimated_probabilities,
            marker="o",
            linestyle="-"
        )

        caption = r"Вибіркове середнє $\overline{p}=$" + \
                    f"{round(np.mean(estimated_probabilities),4)}, " + \
                    r"вибіркова дисперсія $S^2=$" + f"{round(np.var(estimated_probabilities),4)}" + "\n" + \
                    r"Оцінка, що максимізує $P(Y=y)$: $\widehat{p}=$" + f"{round(estimated_probabilities[np.argmax(MLE)],4)}, " + \
                    r"$\widehat{q}=$" + \
                    f"{[round(estimated_distortion_coefficients[np.argmax(MLE)][i],4) for i in range(len(estimated_distortion_coefficients[0]))]}"

        ax1.set_title(r"Значення параметрів $p^*$ від рестарту до рестарту", y=1.01, font=font)
        ax1.grid(True, linestyle='-.')
        ax1.text(
            0.01, -0.3, caption, 
            horizontalalignment='left',
            verticalalignment='center', 
            transform=ax1.transAxes,
            font=font
        )

        ax2.hist(
            estimated_probabilities, 
            weights=(1/len(estimated_probabilities))*np.ones_like(estimated_probabilities)
        )

        ax2.set_title(r"Гістограма значень $p^*$", y=1.01, font=font)
        ax2.grid(True, axis="y", linestyle='-.')

        ax3.remove()
        ax3 = fig.add_subplot(1,3,3, projection='3d')

        q1 = np.array([estimated_distortion_coefficients[i][0] for i in range(len(estimated_distortion_coefficients))])        
        q2 = np.array([estimated_distortion_coefficients[i][1] for i in range(len(estimated_distortion_coefficients))])

        hist, xedges, yedges = np.histogram2d(q1, q2, bins=(20,20))
        xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])

        xpos = xpos.flatten()/2.0
        ypos = ypos.flatten()/2.0
        zpos = np.zeros_like(xpos)

        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]
        dz = hist.flatten()/len(estimated_distortion_coefficients)

        cmap = cm.get_cmap('jet')
        max_height = np.max(dz)
        min_height = np.min(dz)
        rgba = [cmap((k-min_height)/max_height) for k in dz] 

        ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
        ax3.set_title(r"Гістограма значень $q^*$", y=1.05, font=font)
        ax3.set_xlabel(r"$q_1$", labelpad=7, font=font)
        ax3.set_ylabel(r"$q_2$", labelpad=7, font=font)

    elif len(estimated_parameters[-1]) == 2 and len(estimated_parameters[-1][1]) != 2:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13,5), 
            gridspec_kw={
                'width_ratios': [2.5, 1], 
                'bottom': 0.3,
                'left': 0.05,
                'right': 0.95,  
                'hspace': 0.005
            }
        )

        estimated_probabilities = [estimated_parameters[i][0] for i in range(len(estimated_parameters))]
        estimated_distortion_coefficients = [estimated_parameters[i][1] for i in range(len(estimated_parameters))]
        
        ax1.plot(
            [i+1 for i in range(len(estimated_probabilities))], 
            estimated_probabilities,
            marker="o",
            linestyle="-"
        )
        
        caption = r"Вибіркове середнє $\overline{p}=$" + \
                  f"{round(np.mean(estimated_probabilities),4)}, " + \
                  r"вибіркова дисперсія $S^2=$" + f"{round(np.var(estimated_probabilities),4)}" + "\n" + \
                  r"Оцінка, що максимізує $P(Y=y)$: $\widehat{p}=$" + f"{round(estimated_probabilities[np.argmax(MLE)],4)}, " + \
                  r"$\widehat{q}=$" + \
                  f"{[round(estimated_distortion_coefficients[np.argmax(MLE)][i],4) for i in range(len(estimated_distortion_coefficients[0]))]}"

        ax1.set_title(r"Значення параметрів $p^*$ від рестарту до рестарту", y=1.01, font=font)
        ax1.grid(True, linestyle='-.')
        ax1.text(
            0.01, -0.3, caption, 
            horizontalalignment='left',
            verticalalignment='center', 
            transform=ax1.transAxes,
            font=font
        )

        ax2.hist(
            estimated_probabilities, 
            weights=(1/len(estimated_probabilities))*np.ones_like(estimated_probabilities)
        )
        ax2.set_title(r"Гістограма значень $p^*$", y=1.01, font=font)
        ax2.grid(True, axis="y", linestyle='-.')

    plt.savefig("Images/estimated_probabilities.png")
    plt.show()
    plt.close()

def display_x_viterbi_results_by_mismatch_indices(hamming_distances,mismatch_indices,cs_groups,N,T,r):
    """
    Save two plots:
        1) histogram of hamming distances
        2) histogram of mismatch indices by indices like i=0, i=1 etc.

    Parameters
    ----------
    hamming_distances : int array(r,)
                        Hamming distances per each restart
    mismatch_indices  : int array(r,)
                        Mismatch indices per each restart
    cs_groups         : int array
                        Intersected indices according to I 
    N                 : int
                        Dimention of any state vector
    T                 : int
                        Length of a chain  
    r                 : int
                        Number of restarts
    """

    font = {
        "family": "serif",
        "size": 16
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8,5),
        gridspec_kw={
            'left': 0.05,
            'right': 0.95,  
            'hspace':0.01
        }
    )

    ax1.set_title("Розподіли відстаней Хеммінга між станами", y=1.03, font=font)
    ax1.grid(True, axis="y", linestyle='-.')

    ax2.set_title("Розподіли індексів неспівпадіння між станами", y=1.03, font=font)
    ax2.grid(True, axis="y", linestyle='-.')

    bottom = np.zeros(N)
    for j in range(N+1):
        hd_distribution = [hamming_distances[i][j] for i in range(len(hamming_distances))]

        mi_distribution = [mismatch_indices[i][j] for i in range(len(mismatch_indices))]
        mi_distribution = sum(mi_distribution,[])

        if len(hamming_distances) == 1:
            height = np.zeros(N+1)
            height[j] = hd_distribution[0]/100
            ax1.bar(
                [f"$d={i}$" for i in range(N+1)], 
                height, 
                width=0.5,
            )

            print(f"height: {height}")

        elif len(hamming_distances) != 1:
            if sum(hd_distribution) != 0:
                ax1.hist(
                    hd_distribution, 
                    bins=12, 
                    label=f"$d={j}$", 
                    weights=(1/r)*np.ones_like(hd_distribution),
                )
            else:
                ax1.hist([], label=f"$d={j}$")

            print(f"d={j} : M = {np.mean(hd_distribution)}, S^2 = {np.var(hd_distribution)}")

            ax1.legend()
        
        height = [mi_distribution.count(i)/(len(mismatch_indices)*T) for i in range(N)]
        ax2.bar(
            [f"$i={i}$" for i in range(N)], 
            height, 
            width=0.5,
            label=f"$d={j}$",
            bottom=bottom,
        )
        for i in range(N):
            bottom[i] = bottom[i] + height[i]

        ax2.legend()

    ax2.set_ylim(top=max(bottom)+0.06)
    ax2.set_xticks([f"$i={i}$" for i in range(N)])
    ax2.set_xticklabels(
        labels=[f"$i={i}$" for i in range(N)], 
        fontdict={
            "family": "serif",
            "size": 14
        }
    )

    plt.savefig("Images/x_viterbi_results.png")
    plt.close()

def display_table(dataframe,title,figsize,colWidths,rowLabels,colLabels,colColours,cellColours,savename,bbox):
    """
    Save figure of matplotlib.table()
    """
    
    plt.figure(figsize=figsize)
    plt.title(title, y=1.025, fontdict={"family": "Serif", "size": 16, "weight": "bold"})

    table = plt.table(
        cellText = dataframe.values,
        cellLoc="center", 
        colWidths=colWidths,
        rowLabels=rowLabels, 
        colLabels=colLabels,
        colColours=colColours,
        cellColours=cellColours,
        bbox=bbox, 
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    plt.axis("off")

    plt.savefig("Images/" + savename + ".png")
    plt.close()

def display_x_viterbi_results_by_groups_of_mismatch_indices(hamming_distances,mismatch_indices,cs_groups,N,T,r):
    """
    Save three plots:
        1) histogram of hamming distances
        2) histogram of mismatch indices by groups like G0, G1 etc.
        3) image of groups G0, G1 etc.

    Parameters
    ----------
    hamming_distances : int array(r,)
                        Hamming distances per each restart
    mismatch_indices  : int array(r,)
                        Mismatch indices per each restart
    cs_groups         : int array
                        Intersected indices according to I 
    N                 : int
                        Dimention of any state vector
    T                 : int
                        Length of a chain  
    r                 : int
                        Number of restarts
    """

    font = {
        "family": "serif",
        "size": 16
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16,5),
        gridspec_kw={
            'width_ratios': [3, 3, 1], 
            'left': 0.05,
            'right': 0.95,  
            'hspace':0.01
        }
    )

    ax1.set_title("Розподіли відстаней Хеммінга між станами", y=1.03, font=font)
    ax1.grid(True, axis="y", linestyle='-.')

    ax2.set_title("Розподіли індексів неспівпадіння між станами", y=1.03, font=font)
    ax2.grid(True, axis="y", linestyle='-.')

    bottom = np.zeros(len(cs_groups))
    for j in range(N+1):
        hd_distribution = [hamming_distances[i][j] for i in range(len(hamming_distances))]

        mi_distribution = [mismatch_indices[i][j] for i in range(len(mismatch_indices))]
        mi_distribution = sum(mi_distribution,[])

        if len(hamming_distances) == 1:
            height = np.zeros(N+1)
            height[j] = hd_distribution[0]/100
            ax1.bar(
                [f"$d={i}$" for i in range(N+1)], 
                height, 
                width=0.5,
            )

            print(f"height: {height}")

        elif len(hamming_distances) != 1:
            if sum(hd_distribution) != 0:
                ax1.hist(
                    hd_distribution, 
                    bins=12, 
                    label=f"$d={j}$", 
                    weights=(1/r)*np.ones_like(hd_distribution),
                )
            else:
                ax1.hist([], label=f"$d={j}$")

            print(f"d={j} : M = {np.mean(hd_distribution)}, S^2 = {np.var(hd_distribution)}")

            ax1.legend()
                
        height = np.zeros(len(cs_groups))
        for i in range(len(cs_groups)):
            height[i] = sum([mi_distribution.count(j)/(len(mismatch_indices)*T) for j in cs_groups[i]])

        ax2.bar(
            [f"$G_{i}$" for i in range(len(cs_groups))], 
            height, 
            width=0.5,
            label=f"$d={j}$",
            bottom=bottom,
        )
        for i in range(len(cs_groups)):
            bottom[i] = bottom[i] + height[i]

        ax2.legend()

    ax2.set_ylim(top=max(bottom)+0.06)
    ax2.set_xticks([f"$G_{i}$" for i in range(len(cs_groups))])
    ax2.set_xticklabels(
        labels=[f"$G_{i}$" for i in range(len(cs_groups))], 
        fontdict={
            "family": "serif",
            "size": 14
        }
    )

    ax3.axis("off")
    ax3.imshow(cv2.imread("Images/table.png"))

    plt.savefig("Images/x_viterbi_results.png")
    plt.close()

def display_y_viterbi_results_by_mismatch_indices(hamming_distances,mismatch_indices,I,T,r):
    """
    Savestwo plots:
        1) histogram of hamming distances
        2) histogram of mismatch indices by indices like I0, I1 etc.

    Parameters
    ----------
    hamming_distances : int array(r,)
                        Hamming distances per each restart
    mismatch_indices  : int array(r,)
                        Mismatch indices per each restart
    I                 : int array
                        Set of observed indices
    T                 : int
                        Length of a chain  
    r                 : int
                        Number of restarts
    """

    font = {
        "family": "serif",
        "size": 16
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8,5),
        gridspec_kw={
            'left': 0.05,
            'right': 0.95,  
            'hspace': 0.01
        }
    )

    ax1.set_title("Розподіли відстаней Хеммінга \n між спостереженнями", y=1.01, font=font)
    ax1.grid(True, axis="y", linestyle='-.')

    ax2.set_title("Розподіли індексів неспівпадіння \n між спостереженнями", y=1.01, font=font)
    ax2.grid(True, axis="y", linestyle='-.')

    bottom = np.zeros(len(I))
    for j in range(len(I)+1):
        hd_distribution = [hamming_distances[i][j] for i in range(len(hamming_distances))]

        mi_distribution = [mismatch_indices[i][j] for i in range(len(mismatch_indices))]
        mi_distribution = sum(mi_distribution,[])

        if len(hamming_distances) == 1:
            height = np.zeros(len(I)+1)
            height[j] = hd_distribution[0]/100
            ax1.bar(
                [f"$d={i}$" for i in range(len(I)+1)], 
                height, 
                width=0.5,
            )
        elif len(hamming_distances) != 1:
            if sum(hd_distribution) != 0:
                ax1.hist(
                    hd_distribution, 
                    label=f"$d={j}$",
                    bins=12,
                    weights=(1/r)*np.ones_like(hd_distribution),
                )
            else:
                ax1.hist([], label=f"$d={j}$")

            ax1.legend()
        
        height = [mi_distribution.count(i)/(len(mismatch_indices)*T) for i in range(len(I))]
        ax2.bar(
            [f"$I_{i}$" for i in range(len(I))], 
            height, 
            width=0.5,
            label=f"$d={j}$",
            bottom=bottom,
        )
        for i in range(len(I)):
            bottom[i] = bottom[i] + height[i]

        ax2.legend()

    ax2.set_ylim(bottom=0, top=max(bottom)+0.06)
    ax2.set_xticks([f"$I_{i}$" for i in range(len(I))])
    ax2.set_xticklabels(
        labels=[f"$I_{i}$" for i in range(len(I))], 
        fontdict={
            "family": "serif",
            "size": 14
        }
    )

    plt.savefig("Images/y_viterbi_results.png")
    plt.close()

def display_predicted_implicit_indices(list_of_x_real,list_of_x_predicted,estimated_parameters,real_implicit_indices,T0,T,N):

    jaccard_dictionary = {} # dictionary to store results, obtained using Jaccard metric
    square_dictionary = {}  # dictionary to store results, obtained using square metric

    r = len(list_of_x_predicted) # actual number of restarts

    for i in range(len(list_of_x_predicted)):
        estimate_length = ["maximum"] 
        # estimate_length = ["сonsistent", estimated_parameters[i][0]]

        I = estimate_implicit_indices(
            list_of_x_real[i],
            list_of_x_predicted[i],
            real_implicit_indices,
            estimate_length,
            T0,T,N,
            metrics_type="square"
        )

        d = define_distance(real_implicit_indices,I,"weighted Jaccard")
        
        if I in square_dictionary.keys():
            square_dictionary[I][1] += 1
        else:
            square_dictionary[I] = [d, 1]

        I = estimate_implicit_indices(
            list_of_x_real[i],
            list_of_x_predicted[i],
            real_implicit_indices,
            estimate_length,
            T0,T,N,
            metrics_type="weighted Jaccard"
        )

        d = define_distance(real_implicit_indices,I,"weighted Jaccard")
        
        if I in jaccard_dictionary.keys():
            jaccard_dictionary[I][1] += 1
        else:
            jaccard_dictionary[I] = [d, 1]

    square_distances = [square_dictionary[I][0] for I in square_dictionary.keys() for _ in range(square_dictionary[I][1])]

    jaccard_distances = [jaccard_dictionary[I][0] for I in jaccard_dictionary.keys() for _ in range(jaccard_dictionary[I][1])]

    plt.figure(figsize=(9,5))
    plt.title(f"Histogram of {r} restarts", y=1.025, fontdict={"family": "Serif", "size": 16, "weight": "bold"})

    plt.hist(
        [square_distances, jaccard_distances], 
        weights=[
            (1/len(square_distances))*np.ones_like(square_distances), 
            (1/len(jaccard_distances))*np.ones_like(jaccard_distances)
        ], 
        histtype = "bar",
        label = [
            "Square classifier", 
            "Jaccard classifier"
        ],
    )

    plt.xlim(xmin=-0.04)
    plt.xlabel("Weighted Jaccard distance " + r"$d_J$" + " to the real implicit indices", fontdict={"family": "Serif", "size": 14})
    plt.ylabel("Distribution among restarts", fontdict={"family": "Serif", "size": 14})
    plt.legend(prop = {"family": "Serif", "size": 14})

    plt.savefig("Images/implicit_indices_histogram.png")
    plt.close()
    
    tabulated_data = {}

    classes = set(sum([list(square_dictionary.keys()),list(jaccard_dictionary.keys())],[]))
    classes_quality = [define_distance(real_implicit_indices,I,"weighted Jaccard") for I in classes]

    for d,I in sorted(zip(classes_quality,classes)):
        if (I in square_dictionary.keys()) and (I in jaccard_dictionary.keys()):
            tabulated_data[f"{I}"] = [
                d, 
                square_dictionary[I][1]/r, 
                jaccard_dictionary[I][1]/r, 
                np.average([square_dictionary[I][1]/r, jaccard_dictionary[I][1]/r], weights=[1/2,1/2]),
                np.average([square_dictionary[I][1]/r, jaccard_dictionary[I][1]/r], weights=[14/39,25/39]),
            ]
        elif I in square_dictionary.keys():
            tabulated_data[f"{I}"] = [
                d, 
                square_dictionary[I][1]/r, 
                0.0, 
                np.average([square_dictionary[I][1]/r, 0.0], weights=[1/2,1/2]),
                np.average([square_dictionary[I][1]/r, 0.0], weights=[14/39,25/39]),
            ]
        elif I in jaccard_dictionary.keys():
            tabulated_data[f"{I}"] = [
                d, 
                0.0, 
                jaccard_dictionary[I][1]/r,
                np.average([0.0, jaccard_dictionary[I][1]/r], weights=[1/2,1/2]),
                np.average([0.0, jaccard_dictionary[I][1]/r], weights=[14/39,25/39]),
            ]    

    tabulated_dataframe = pd.DataFrame(tabulated_data, index=[r"Distance $d_J$", "Square classifier", "Jaccard classifier", "Voting classifier\n    (0.5 / 0.5)", "Voting classifier\n  (0.36 / 0.64)"]).round(2)

    col_colors = None
    cell_colors = [["white" for j in range(len(tabulated_dataframe.columns))] for i in range(len(tabulated_dataframe.index))]
    cell_colors[tabulated_dataframe.index.get_loc("Voting classifier\n    (0.5 / 0.5)")][np.argmax(tabulated_dataframe.loc["Voting classifier\n    (0.5 / 0.5)"])] = "honeydew"
    cell_colors[tabulated_dataframe.index.get_loc("Voting classifier\n  (0.36 / 0.64)")][np.argmax(tabulated_dataframe.loc["Voting classifier\n  (0.36 / 0.64)"])] = "honeydew"

    display_table(
        tabulated_dataframe,
        title="Tabulated data",
        figsize=(9,5),
        colWidths=[(1-0.1)/len(tabulated_dataframe.columns) for _ in range(len(tabulated_dataframe.columns))],
        rowLabels=tabulated_dataframe.index,
        colLabels=tabulated_dataframe.columns,
        colColours=col_colors,
        cellColours=cell_colors,
        savename="implicit_indices_table",
        bbox=[0.2, 0, 0.9, 1]
    )

    implicit_indices_table = cv2.imread("Images/implicit_indices_table.png")
    implicit_indices_histogram = cv2.imread("Images/implicit_indices_histogram.png")

    implicit_indices_figure = np.concatenate((implicit_indices_histogram, implicit_indices_table), axis=1)

    cv2.imwrite("Images/implicit_indices_figure.png", implicit_indices_figure)

    return implicit_indices_figure, tabulated_dataframe, tabulated_dataframe.keys()[np.argmax(tabulated_dataframe.loc["Voting classifier\n    (0.5 / 0.5)"])]