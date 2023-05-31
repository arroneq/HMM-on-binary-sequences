from __future__ import annotations    # enable using function specifications
import numpy.typing as npt        
import typing                      

import numpy as np
import itertools
import copy

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
    alpha : float array(pow(2,N),T)
        Forward algorithm coefficients
    P : float
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
        scaling: bool
    ):
    """
    Return learning algorithm results (estimated parameters)

    Parameters
    ----------
    y : int array(T,)
        Chain of observations
    
    N : int
        Dimention of any state vector 
    T : int
        Length of a chain
    I : int array
        Set of observed indices
    estimator : string
        Type of estimation (only p or both p & q estimation)    
    p0 : float
        Initial approximation of parameter p 
    q0 : float array(len(I),)
        Initial approximation of distortion coefficients q 
    scaling : string
        Either to scale forward and backward coefficients or not

    Returns
    -------
    parameter : float array
        List of estimated parameters [p,q] for each iteration
    joint_probabilities : float array
        List of P(Y=y) probabilities (concerns to forward algorithm) 
    joint_probabilities_increments : float array
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

    number_of_iterations = 0

    while (
        number_of_iterations < 20 or 
        abs(1-joint_probabilities[-1]/joint_probabilities[-2]) > 0.0001
    ):
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
            joint_probabilities_increments.append(
                abs(copy.deepcopy(P) - joint_probabilities_increments[-1])
            )

        p_numerator = calculate_p_numerator(y,alpha,beta,A,B,N,T)
        p_denominator = calculate_p_denominator(alpha,N,T)
        p = p_numerator/p_denominator

        parameter.append([copy.deepcopy(p)])

        if estimator == "parameter p and coefficients q estimation task":
            for j in range(len(I)):
                part_1 = calculate_q_estimation_part_1(y,q[j],alpha,beta,A,N,T,I,j)
                part_2 = calculate_q_estimation_part_2(y,q[j],alpha,beta,A,N,T,I,j)

                qj_numerator = copy.deepcopy(part_1)
                qj_denominator = copy.deepcopy(part_1) + copy.deepcopy(part_2)

                q[j] = qj_numerator/qj_denominator

            parameter[number_of_iterations + 1].append(copy.deepcopy(q))

        number_of_iterations += 1       

    return parameter, joint_probabilities, joint_probabilities_increments

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

def define_distance(
        A: npt.NDArray[np.float64], 
        B: npt.NDArray[np.float64], 
        label: typing.Literal["square", "weighted Jaccard"]
    ):
    """ 
    Return a distance between set A and set B
    """

    if label == "square":
        return sum([pow(a-b,2) for a,b in zip(A,B)])
    if label == "weighted Jaccard":
        return 1 - sum([min(a,b) for a,b in zip(A,B)])/sum([max(a,b) for a,b in zip(A,B)])

def estimate_implicit_indices(
        x_real: list[str],
        x_predicted: list[int],
        real_implicit_indices: list[int],
        estimate_length: typing.Literal["maximum", "сonsistent"],
        T: int,
        N: int,
        metrics_type: typing.Literal["square", "weighted Jaccard"]
    ) -> tuple[int]:
    """
    Return estimation of implicit indices in a hidden chain 

    Parameters
    ----------
    x_real : string array(T,)
        Real chain of the hidden states
    x_predicted : int array(T,)
        Decoded enumarated state chain
    real_implicit_indices : int array
        Real set of implicit indices
    estimate_length : string array()
        A way to estimate length of set of implicit indices: 
        eigher by "maximum" method or by using "consistent" estimation
    T : int
        Length of a chain
    N : int
        Dimention of any state vector 
    metrics_type : string
        Either "square" or "weighted Jaccard"

    Returns
    -------
    predicted_implicit_indices : int array(T,)
        Set of predicted_implicit_indices 
        (considering an acceptable list of "0", "1", "2" etc. for given combination) 
    """

    phi_real = np.zeros(T, dtype=int)
    for t in range(T):
        phi_real[t] = sum([int(list(x_real[t])[i]) for i in real_implicit_indices])   

    if estimate_length[0] == "maximum":
        estimated_length = max(phi_real)
    elif estimate_length[0] == "consistent":
        p = estimate_length[1]
        estimated_length = int(
            (N/(1-p))*(1 - sum([1 for t in range(T-1) if phi_real[t] == phi_real[t+1]])/(T-1))
        )

    offered_implicit_indices = list(itertools.combinations([i for i in range(N)], estimated_length))

    metric = np.zeros(len(offered_implicit_indices))
    for k in range(len(offered_implicit_indices)):
        phi_offered = np.zeros(T, dtype=int)
        for t in range(T):
            phi_offered[t] = sum([int(list(x_predicted[t])[i]) for i in offered_implicit_indices[k]])
        
        metric[k] = define_distance(phi_real,phi_offered,metrics_type)

    min_metric_value = min(metric)
    argmin_metric_value = [
        index for index in range(len(metric)) if metric[index] == min_metric_value
    ]

    predicted_implicit_indices = []
    for index in argmin_metric_value:
        predicted_implicit_indices.append(offered_implicit_indices[index])

    return predicted_implicit_indices[0]