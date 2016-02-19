import pdb
import random
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator
import operator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, trials=1):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.learning_rate = 0.9
        self.Q = {}

        # Standard default Q value of 0
        # self.default_Q = 0

        # Optimal default Q value.
        self.default_Q = 1

        # discount factor is denoted by Beta in official Bellman equation formula
        # (http://web.stanford.edu/~pkurlat/teaching/5%20-%20The%20Bellman%20Equation.pdf),
        # or gamma in Udacity course.
        self.discount_factor = 0.33
        
        # How likely are we to pick random action / explore new paths?
        self.epsilon = 0.1 

        # How many times agent able to reach the target for given trials?
        self.success = 0
        self.total = 0
        self.trials = trials

        # How many penalties does an agent get?
        self.penalties = 0
        self.moves = 0

        self.net_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        inputs['waypoint'] = self.next_waypoint

        # Tried deleting some inputs to see how it affected performance.
        # Success rate did improve significantly, and counterintuitively, less penalty.
        del inputs['oncoming']
        del inputs['left']
        del inputs['right']

        # Tried to see if deadline info would improve performance turns out it got worse.
        # inputs['deadline'] = deadline

        self.state = tuple(sorted(inputs.items()))

        # TODO: Select action according to your policy

        # Random action 
        # action = random.choice(Environment.valid_actions)

        # Q learning chosen action
        _Q, action = self._select_Q_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Some stats
        self.net_reward += reward
        self.moves += 1
        if reward < 0:
            self.penalties+= 1

        add_total = False
        if deadline == 0:
            add_total = True
        if reward > 5:
            self.success += 1
            add_total = True
        if add_total:
            self.total += 1
            print self._more_stats()

        # TODO: Learn policy based on state, action, reward

        # Note that we are updating for the previous state's Q value since Utility function is always +1 future state.
        if self.prev_state != None:
            if (self.prev_state, self.prev_action) not in self.Q:
                self.Q[(self.prev_state, self.prev_action)] = self.default_Q

            # Update to Q matrix as described in this lesson:
            # https://www.udacity.com/course/viewer#!/c-ud728-nd/l-5446820041/m-634899057            
            self.Q[(self.prev_state,self.prev_action)] += \
            self.learning_rate * (self.prev_reward + self.discount_factor * \
                self._select_Q_action(self.state)[0] - self.Q[(self.prev_state, self.prev_action)])

        # pdb.set_trace()
        self.prev_state = self.state
        self.prev_action = action
        self.prev_reward = reward

        self.env.status_text += ' ' + self._more_stats()

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def _more_stats(self):
        """Get additional stats"""
        return "success/total = {}/{} of {} trials (net reward: {})\npenalties/moves (penalty rate): {}/{} ({})".format(
                self.success, self.total, self.trials, self.net_reward, self.penalties, self.moves, round(float(self.penalties)/float(self.moves), 2))

    def _select_Q_action(self, state):
        """Select max Q and action based on given state.

        Args:
            state(tuple): Tuple of state e.g. (('heading', 'forward'), ('light', 'green'), ...).

        Returns:
            tuple: max Q value and best action
        """
        best_action = random.choice(Environment.valid_actions)
        if self._random_pick(self.epsilon):
            max_Q = self._get_Q(state, best_action)
        else:
            max_Q = -999999
            for action in Environment.valid_actions:
                Q = self._get_Q(state, action)
                if Q > max_Q:
                    max_Q = Q
                    best_action = action
                elif Q == max_Q:
                    if self._random_pick(0.5):
                        best_action = action
        return (max_Q, best_action)


    def _get_Q(self, state, action):
        """Get Q value given state and action.

        If Q value not found in self.Q, self.default_Q will be returned instead.

        Args:
            state(tuple): Tuple of state e.g. (('heading', 'forward'), ('light', 'green'), ...).
            action(string): None, 'forward', 'left', or 'right'.
        """
        return self.Q.get((state, action), self.default_Q)

    def _random_pick(self, epsilon=0.5):
        """Random pick with epsilon as bias.

        The larger epsilon is, the more likely it is to pick random action.

        Explanation is available at this course:
        https://www.udacity.com/course/viewer#!/c-ud728-nd/l-5446820041/m-634899065 starting at 2:10.

        One may consider this function as: When True, do random action.

        For equal chance, use epsilon = 0.5

        Args:
            epsilon(float): Likelihood of True.
        """
        return random.random() < epsilon

def run():
    """Run the agent for a finite number of trials."""

    trials = 100
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, trials)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.00000001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=trials)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
