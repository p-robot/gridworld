#!/usr/bin/env python
"""
Module of utilities
"""

import numpy as np


def egreedy(Qs, actions, epsilon):
    """
    Epsilon greedy policy.  
    
    This function represents an epsilon greedy policy used to choose actions
    given a state.  The action with the maximum value in the value function
    is chosen, but a random action is chosen with probability epsilon (an 
    attribute of the agent).  Ties in maximum value function between 
    different actions are resolved randomly.  
    
    Parameters
    ---------
    Qs: np.array
        Current value function
    
    actions : list
        List of actions
    
    epsilon : double
        Learning parameter
    
    Returns
    -------
    The action to take given the current state according to an egreedy algorithm.  
    """
    
    # Find the action with the maximum value
    action = np.where(Qs == np.max(Qs))[0]
    
    # Deal with ties 
    if len(action) > 1:
        np.random.shuffle(action)
    
    if len(action) == 0:
        action = [0]
    
    action = action[0]
    
    # Choose a random action with probability epsilon
    if np.random.rand() < epsilon:
        action = np.random.randint(len(actions))
    
    return actions[action]

