#!/usr/bin/env python3
"""
Windy gridworld example (example 6.5) from Sutton and Barto (1998)
using a dictionary to store the value function.  

This RL task represents a gridded world where an agent has to move from a 
starting state to a goal state.  In this particular example, there is wind 
in the environment so that the agent is pushed upwards in particular positions.
The agent can move up, down, left or right, depending on the limits of the 
gridded world, and the effects of wind.  This example can be run by first 
importing the module using `from am4fmd.examples import gridworld_tablular` 
and then calling the function `gridworld_tablular.run()`.  

This is different to the example in am4fmd.examples.main_gridlworld_ex65 in 
that it uses a dictionary to store the value function instead of using a numpy
array.  

W. Probert, 2014
"""

import copy, numpy as np
from os.path import join
from matplotlib import pyplot as plt
import rli, utils
import random


class MCAgent(rli.Agent):
    """
    Class representing an agent that learns using epsilon-soft Monte Carlo control
    
    If no starting action is given then a random action is chosen.  
    """
    def __init__(self, actions, epsilon, control_switch_times = range(300001), \
            starting_action = None, verbose = False):
        
        self._verbose = verbose
        self._epsilon = epsilon
        self._control_switch_times = control_switch_times
        self._starting_action = starting_action
        
        # List all valid actions
        self._actions = actions
        
        self.Q = dict()
        self.visits = dict()
        self.sa_seen = set()
        
        # List of durations resulting from each state-action pair.  
        # Dict keys are states; then each value is a multi-dimensional numpy array
        self.returns = dict()
    
    def start_trial(self, state):
        """
        Return starting action at the start of the trial
        """
        
        # Empty the list of seen state-action pairs
        self.sa_seen = set()
        
        if self.starting_action is None:
            action = random.choice(self.actions)
        else:
            action = self.starting_action
        
        # If Q[s] has not been seen before, create a table for it.
        if not(state in self.Q):
            self.Q[state] = np.empty(len(self.actions))
            self.Q[state][:] = 0.0
            self.visits[state] = 0
            
            # Create a numpy array to which we can append outbreak durations
            self.returns[state] = [[] for a in self.actions]
        
        return action
    
    def step(self, s, action, reward, next_s, t, *args):
        
        # Check for terminal state
        if (next_s == self._terminal_state):
            
            # Return a dummy action
            next_a = 20
            out_action = next_a
            
            # Loop through all state action pairs that we've seen, and append the return
            # and update the Q dictionary.  
            for s1, a1 in self.sa_seen:
                
                # Record the observed state
                self.visits[s1] +=1
                
                # Duration of the outbreak is the variable t, append t to the returns[] list
                self.returns[s1][a1].append(t)
                
                # Find the average of the durations for the state-action pair visited
                self.Q[s1][a1] = np.mean(self.returns[s1][a1])
            
        else:
            # Check that it's time to change the action
            if t in self.control_switch_times:
                
                # Convert action to an index
                ind = [i for i, v in enumerate(self.actions) if v == action][0]
                
                # Add the current state and action to the set of seen states
                self.sa_seen.add((s, ind))
                
                # If Q[s] has not been seen before, create a table for it.
                if not(s in self.Q):
                    self.Q[s] = np.empty(len(self.actions))
                    self.Q[s][:] = 0.0
                    self.visits[s] = 0
                    
                    # Create a numpy array to which we can append outbreak durations
                    self.returns[s] = [[] for a in self.actions]
                
                if not(next_s in self.Q):
                    self.Q[next_s] = np.empty(len(self.actions))
                    self.Q[next_s][:] = 0.0
                    self.visits[next_s] = 0
                    
                    # Create a numpy array to which we can append outbreak durations
                    self.returns[next_s] = [[] for a in self.actions]
                
                # Determine the list of probabilities of choosing an action
                action_probabilities = utils.epsilon_soft(self.Q[s], self.actions, self.epsilon)
                
                next_a_idx = np.random.choice(len(self.actions), 1, p = action_probabilities)[0]
                next_a = self.actions[next_a_idx]
                
            else:
                next_a = action
        
        return next_a
    
    @property
    def verbose(self):
        "Should verbose output be turned on?"
        return self._verbose
    
    @property
    def epsilon(self):
        "Epsilon parameter"
        return self._epsilon
    
    @property
    def actions(self):
        "Actions"
        return self._actions
    
    @property
    def control_switch_times(self):
        "List of times at which control can switch"
        return self._control_switch_times
    
    @property
    def starting_action(self):
        "Starting action"
        return self._starting_action


class SarsaAgent(rli.Agent):
    """
    Class representing a SARSA Agent
    
    This class represents an agent that learns using the SARSA RL algorithm.  
    See Sutton and Barto (1998) for the section on temporal difference 
    learning and the defintion of the SARSA algorithm.  This implementation 
    does not include eligibility traces and uses a tabular function 
    approximation (the Q-table is stored in a dictionary that has keys of 
    states and a vector of values, one for each action, within those).  
    
    This class is similar to the SarsaAgent class within 
    am4fmd.examples.fmd_culling except it uses the egreedy function from within
    the am4fmd.utils module (instead of as a method of the class) and does not
    store the 'ind' property of actions used to refer to the action index.  
    
    Attributes
    -----------
    epsilon:  the probability of choosing a random action at each decision point
    alpha: the step-size parameter (see Sutton and Barto (1998))
    gamma: the discount rate for rewards
    Q: the value function, here a dictionary
    trace: a dictionary recording the number of visits to each state
    actions: a list of actions that can be taken at any decision point.
    
    Methods
    -------
    start_trial: Ready the agent for the start of a trial by returning a 
        first action
    step: Step the agent through a timestep of the simulation
    """
    def __init__(self, actions, epsilon, alpha, gamma):
        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma
        self.Q = dict()
        self.visits = dict()
        
        # List all valid actions
        self._actions = actions
    
    def start_trial(self, state):
        """
        Ready the agent for the start of a trial.  
        
        This method uses an epsilon-greedy algorithm, the current value 
        function, and the current state to choose the first action in a 
        simulation.  If the state has not been seen before then an array of 
        large negative values are used to initialize the value of each action 
        in that state.  This is so that the algorithm is forced to sample all
        the actions, for this same state, through simulation time.  
        
        Parameters
        ----------
        state: 
            the starting state of the environment
        
        Returns
        -------
        action: 
            The action to take given being in state 'state'.  
        """
        
        # If Q[state] has not been seen before, create an array for it.  
        if not(state in self.Q):
            self.Q[state] = np.zeros(len(self.actions))
            self.visits[state] = 0
        
        action = utils.egreedy(self.Q[state], self.actions, self.epsilon)
        
        return action
    
    def step(self, s, action, reward, next_s, t, *args):
        """
        Run the agent through one timestep of the simulation.  
        
        This is where learning occurs.  See the step method of the Agent class
        in the rli.py module for a more general description of this method.  
        At each time step of the simulation the agent checks if the terminal 
        state has been reched.  If so, a dummy action is returned.  Otherwise 
        the agent checks to see if the state and next state have been observed 
        before, if they haven't then the Q value function and trace dictionaries 
        are updated with empty arrays.  Finally, the next action is chosen 
        using an epsilon greedy algorithm and the value function is updated 
        using the sarsa algorithm.  The trace dictionary records how many times 
        each state has been observed.  The next action to take is returned 
        (as an action object, not an index).  
        
        Parameters
        ----------
        s :
            current state
        action : 
            action taken in state s
        reward : float/int
            reward from taking the action in state s
        next_s :
            next state
        t : int
            current time step, to check if this is a decision horizon
            
        Returns
        -------
        out_action : 
            The next action to be taken.  This is an action object, not an index.
            
        """
        
        # Convert action to an index
        action = [i for i, v in enumerate(self.actions) if v == action][0]
        
        # Check for terminal state
        if (next_s == self._terminal_state):
            # Return a dummy action
            next_a = 20
            out_action = next_a
        else:
            # If Q[s] has not been seen before, create a table for it.
            if not(s in self.Q):
                self.Q[s] = np.empty(len(self.actions))
                self.Q[s][:] = 0.0
                self.visits[s] = 0
            
            if not(next_s in self.Q):
                self.Q[next_s] = np.empty(len(self.actions))
                self.Q[next_s][:] = 0.0
                self.visits[next_s] = 0
            
            # Find the next action
            next_a = utils.egreedy(self.Q[next_s], self.actions, self.epsilon)
            next_a = [i for i, v in enumerate(self.actions) if v == next_a][0]
            out_action = self.actions[next_a]
            # Update the Q function
            target = reward + self.gamma * self.Q[next_s][next_a]
            
            self.Q[s][action] = \
                self.Q[s][action] + \
                self.alpha*(target - self.Q[s][action])
            
            # Record the observed state
            self.visits[s] +=1
            
        return out_action
    
    @property
    def epsilon(self):
        "Epsilon parameter"
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, value):
        "Allow epsilon to be set, to avoid an AttributeError"
        self._epsilon = value
    
    @property
    def gamma(self):
        "Gamma parameter"
        return self._gamma
    
    @property
    def alpha(self):
        "Alpha parameter"
        return self._alpha
    
    @property
    def actions(self):
        "Actions"
        return self._actions


class SarsaAgentTraces(rli.Agent):
    """
    """
    def __init__(self, actions, epsilon, alpha, gamma, lamb, tracing = "replace"):
        
        self._epsilon = epsilon
        
        self._alpha = alpha
        self._gamma = gamma
        self.Q = dict()
        self.visits = dict()
        self.Z = dict()
        self.sa_seen = set()
        
        self._lamb = lamb
        self.tracing = tracing
        
        # List all valid actions
        self._actions = actions
        
    def start_trial(self, state):
        """
        """
        
        # If Q[state] has not been seen before, create an array for it.  
        if not(state in self.Q):
            self.Q[state] = np.zeros(len(self.actions))
            self.visits[state] = 0
        
        # Reset the set of seen states, and the traces
        self.sa_seen = set()
        self.Z = dict()
        self.Z[state] = np.zeros(len(self.actions))
        
        action = utils.egreedy(self.Q[state], self.actions, self.epsilon)
        
        return action
    
    def step(self, s, action, reward, next_s, t, *args):
        """
        """
        # Convert action to an index
        action = [i for i, v in enumerate(self.actions) if v == action][0]
        
        # Check for terminal state
        if (next_s == self._terminal_state):
            # Return a dummy action
            next_a = None
            out_action = next_a
        else:
            # Add the current state and action to the set of seen states
            self.sa_seen.add((s,action))
            
            # If Q[s] has not been seen before, create a table for it.
            if not(s in self.Q):
                self.Q[s] = np.empty(len(self.actions))
                self.Q[s][:] = 0.0
                self.visits[s] = 0
            if not(s in self.Z):
                self.Z[s] = np.empty(len(self.actions))
                self.Z[s][:] = 0.0
            
            if not(next_s in self.Q):
                self.Q[next_s] = np.empty(len(self.actions))
                self.Q[next_s][:] = 0.0
                self.visits[next_s] = 0
            if not(next_s in self.Z):
                self.Z[next_s] = np.empty(len(self.actions))
                self.Z[next_s][:] = 0.0
            
            # Find the next action
            next_a = utils.egreedy(self.Q[next_s], self.actions, self.epsilon)
            next_a = [i for i, v in enumerate(self.actions) if v == next_a][0]
            out_action = self.actions[next_a]
            
            delta = reward + self.gamma * self.Q[next_s][next_a] - \
                self.Q[s][action]
            
            # Update Q function and Z traces
            for si, ai in self.sa_seen:
                if self.tracing == "replace":
                    self.Z[si][ai] = np.minimum(self.gamma * self.lamb * self.Z[si][ai], 1)
                else:
                    self.Z[si][ai] = self.gamma * self.lamb * self.Z[si][ai]
            
            # Udpate the traces
            self.Z[s][action] += 1
            
            # Update Q function and Z traces
            for si, ai in self.sa_seen:
                self.Q[si][ai] = self.Q[si][ai] + \
                    self.alpha * delta * self.Z[si][ai]
            
            # Record the observed state
            self.visits[s] += 1
        
        return out_action
    
    @property
    def epsilon(self):
        "Epsilon parameter"
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, value):
        "Allow epsilon to be set, to avoid an AttributeError"
        self._epsilon = value
    
    @property
    def lamb(self):
        "Lambda parameter"
        return self._lamb
    
    @property
    def gamma(self):
        "Gamma parameter"
        return self._gamma
    
    @property
    def alpha(self):
        "Alpha parameter"
        return self._alpha
    
    @property
    def actions(self):
        "Actions"
        return self._actions


class GridworldEnv(rli.Environment):
    """
    Class representing windy gridworld environment
    
    This class represents the gridworld environment including the starting and 
    goal states of the agent.  Constraints on movements, such as the effects of
    walls and wind are documented here.  
    
    Attributes
    ----------
        states: tuples of coordinates of all possible states in the gridworld
        xlim: length of the gridworld in the x direction
        ylim: length of the gridworld in the y direction
        start: starting state in the gridworld
        goal: goal state in the gridworld
        actions: list of available actions, as increments to the x and y coords
            of the current state
        
    Methods
    -------
        step: run the environment through a timestep of the simulation
        wind_effect: generate the effect of wind on the state
        edge_effect: generate the effect of edges of the gridworld on the state
        See the rli module for a list of methods of an environment object
    """
    def __init__(self, xlim = 10, ylim = 7, start = (0,3), goal = (7,3)):
        
        x = range(xlim); y = range(ylim)
        X, Y = np.meshgrid(x, y)
        self._states = zip(X.flatten(), Y.flatten())
        self.xlim = xlim
        self.ylim = ylim
        
        # Starting state and goal state
        self._start = start
        self._goal = goal
    
    def step(self, state, action, t):
        """
        Run the environment through one time step of the simulation.  
        
        This method performs the given action in the current state.  Both the
        wind and edge effect methods are then applied to the state.  A reward 
        of -1 is given at each time step that's not a transition to the 
        terminal state.  See rli.py for generic method descriptions from the 
        rli module.  
        
        Args:
            state: current state
            action: action taken in the current state
        
        Returns:
            reward, and the next state
        """
        next_s = tuple(sum(x) for x in zip(state, action))
        # Apply wind effect, apply edge effect
        next_s = self.wind_effect(next_s)
        next_s = self.edge_effect(next_s)
        
        reward = -1
        
        # The special value for the terminal state.  
        if next_s == self.goal:
            next_s = self._terminal_state
            reward = 0
        
        return reward, next_s
        
    def wind_effect(self, s):
        """
        Generate the effect of wind on the state.  
        
        Following the windy gridworld example of Sutton and Barto, this method
        emulates the effect of wind on the state.  If the agent is in position
        3, 4, 5, or 8 horizontally, then the agent is moved up by one square.  
        If the agent is in position 6 or 7 then the agent is moved up by two 
        squares by the wind.  
        
        Args:
            s: current state of the system
        
        Returns: 
            State after the effect of wind has been accounted for.  
        """
        
        if s[0] in [3, 4, 5, 8]:
            wind = [0, 1]
        elif s[0] in [6, 7]:
            wind = [0, 2]
        else:
            wind = [0, 0]
        new_state = (s[0] + wind[0], s[1] + wind[1])
        return(new_state)
    
    def edge_effect(self, state):
        """
        Emulate the effect of edges/walls in the gridworld environment.  
        
        The gridworld environment is a limited environment.  Therefore there 
        must be a mechanism for keeping the agent inside the walls of the 
        gridworld.  This method takes a state and makes sure that the action
        taken doesn't push the agent outside of the gridworld environment.  If
        any state is beyond the limits of the gridworld it is clipped back to
        be within the bounds of the walls.  If the x value is less than 0 then 
        it is set to zero, if the x value is greater than 9 then it is set to
        9, if the y value is less than 0 then it is set to 0, and if it is 
        greater than 6 then it is set to 6.  An alternative approach to 
        limiting the movement of the agent in the environment would be to make 
        actions that could push the agent outside the gridworld environment 
        impossible to the agent in particular states.  
        
        Arguments
        ---------
        state: 
            The current state of the environment
        
        Returns
        -------
        new_state
            State of the environment after accounting for the limited size of 
            the gridworld environment (clipping to the inside region)
        """
        
        new_state = state
        if state[0] < 0:
            new_state = (0, new_state[1])
        if state[0] > (self.xlim - 1):
            new_state = ((self.xlim - 1), new_state[1])
        if state[1] < 0:
            new_state = (new_state[0], 0)
        if state[1] > (self.ylim - 1):
            new_state = (new_state[0], (self.ylim - 1))
        return(new_state)
    
    @property
    def states(self):
        "List of states"
        return self._states
    
    @property
    def goal(self):
        "Goal state"
        return self._goal


class WindyGridlworldSim(rli.Simulation):
    """
    Windy grid world simulation object.  
    
    See the Simulation object in the rli module for a description of the 
    methods.  The only custom method in this subclass is collect_data.  
    
    
    Parameters
    ----------
    Agent : rli.Agent object
        Agent object to be used in the simulation
    Environment : rli.Environment object
        Environment object to be used in the simulation
    max_time: float
        maximum walltime (in hrs) for all trials or timesteps to run.
        The simulation stops if the max_time is reached.  
    
    Attributes
    ----------
    agt: rli.Agent object
        agent to be used in the simulation
    env: environment to be used in the simulation
    max_time: float
        the maximum wall time (in hours) for any simulation
    
    outlist: list
        a list that saves all states, rewards, and actions in the simulation so far
    durations: list 
        a list of durations of each simulation
    current_s: rli.
        the current state in the simulation
    current_a: 
        the current action in the simulation
    terminal_state: 
        the placeholder for the terminal state
    
    Methods
    --------
        collect_data: save data on states/actions/rewards throughout the simulation
    
    Further methods are defined in the rli.Simulation parent class.  
    """
    
    def __init__(self, Agent, Environment, max_time, **kwargs):
        
        # Call the initialisation method of the parent class (in this case rli.Simulation)
        super(self.__class__, self).__init__(Agent = Agent, Environment = Environment, 
            max_time = max_time, **kwargs)
        
        self.agt = Agent
        self.env = Environment
        self.max_time = max_time
        
        self.current_trial = []
        self.outlist = []
        self.durations = []
        
        self.current_s = None
        self.current_a = None
        
        self._terminal_state = "terminal"
        self.start_trial()


def run():
    """
    Run the windy gridworld example comparing with/without eligibility traces.
    
    This demo creates a windy gridworld environment, creates a sarsa agent 
    without eligibility traces, an agent with eligibility traces, and a 
    simulation object to manage the simulation.  The demo then runs the windy
    gridworld example for 170 simulations (as per the diagram for the windy 
    gridworld within Sutton and Barto).  Plots for the number of timesteps per
    episode are printed for both agents.  Finally, the optimal actions are 
    printed to the screen, and a plot of simulation durations versus simulation
    number is plotted (as per Sutton and Barto).  Alternatively, users could
    run the trial for 8000 time steps (which should end in a similar position).
    """
    
    seed = 2020
    print("Seed used:", seed)
    np.random.seed(seed)
    
    # Create the gridworld environment
    wgw = GridworldEnv()
    
    # Define list of actions and agent (without eligibilty traces)
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Define a Monte Carlo agent
    agent_mc = MCAgent(actions, epsilon = 0.1)
    
    # Define a SARSA agent
    agent_wo = SarsaAgent(actions, epsilon = 0.1, alpha = 0.5, gamma = 1.0)
    
    # Define agent with eligibility traces (replacing)
    agent_re = SarsaAgentTraces(actions, epsilon = 0.1, alpha = 0.5, gamma = 1.0, \
        lamb = 0.7, tracing = 'replace')
    
    # # Define agent with eligibility traces (accumulating)
    agent_ac = SarsaAgentTraces(actions, epsilon = 0.1, alpha = 0.5, gamma = 1.0, \
        lamb = 0.7, tracing = 'accumulating')
    
    # Create the simulation
    sim_mc = WindyGridlworldSim(agent_mc, wgw, np.inf)
    sim_wo = WindyGridlworldSim(agent_wo, wgw, np.inf)
    sim_re = WindyGridlworldSim(agent_re, wgw, np.inf)
    sim_ac = WindyGridlworldSim(agent_ac, wgw, np.inf)
    
    # Run the simulation for 170 episodes (compare with figure 6.11 of Sutton and Barton)
    N = 170 # use 500 with MCAgent (and expand the x-y plotting limits
    sim_mc.trials(N, max_steps_per_trial = 300000)
    sim_wo.trials(N)
    sim_re.trials(N)
    sim_ac.trials(N)
    
    durations_mc = np.insert(np.cumsum(sim_mc.durations), 0, 0)
    durations_wo = np.insert(np.cumsum(sim_wo.durations), 0, 0)
    durations_re = np.insert(np.cumsum(sim_re.durations), 0, 0)
    durations_ac = np.insert(np.cumsum(sim_ac.durations), 0, 0)
    
    # Plotting functions
    fig, ax = plt.subplots()
    
    # If the quickest path is 15, this is the slope of the optimal ("quickest") path from 
    # starting state to goal state
    opt = 15
    for i in np.arange(-2000, 8000, 500):
        ax.plot([0+i, opt*N+i], [0, N], c = 'grey', linestyle = 'dotted',
            alpha = 0.7, label = "", linewidth = 0.75)

    # Plot once again with label added so it's picked up by the legend command
    ax.plot([0+i, opt*N+i], [0, N], c = 'grey', linestyle = 'dotted',
        alpha = 0.7, linewidth = 0.75, label = "Quickest path")
    
    ax.plot(durations_mc, range(len(durations_mc)), label = "MC control", linewidth = 2)
    ax.plot(durations_wo, range(len(durations_wo)), label = "Without", linewidth = 2)
    ax.plot(durations_re, range(len(durations_re)), label = "Replacing", linewidth = 2)
    ax.plot(durations_ac, range(len(durations_ac)), label = "Accumulating", linewidth = 2)
    
    ax.set_xlim([0, 8000]); ax.set_ylim([0, N])
    ax.set_xlabel("Time steps"); ax.set_ylabel("Episode number")
    
    ax.set_title("Comparison of learning rate with/without eligibility traces\n \
        for the windy gridworld example of Sutton and Barto (ex 6.5)")
    
    plt.legend(loc = "lower right")
    plt.savefig(join("graphics", "gridworld_comparison_seed_" + str(seed) + ".png"), dpi = 300)
    plt.close()


if __name__ == "__main__":
    """
    If this script is called from the command-line then call the run() function.  
    """
    run()
