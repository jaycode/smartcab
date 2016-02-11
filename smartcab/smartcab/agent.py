import pdb
import random
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator
import operator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.learning_rate = 0.8
        self.Q = {}
        self.total_penalties = 0
        self.total_moves = 0

        for light in TrafficLight.valid_states:
            if light == True:
                light = 'red'
            else:
                light = 'green'
            for oncoming_action in Environment.valid_actions:
                for left_action in Environment.valid_actions:
                    for right_action in Environment.valid_actions:
                        for waypoint_direction in Environment.valid_actions:
                            # Flatten dict into a tuple so it can be used as key.
                            key = tuple(sorted({
                                'light':light,
                                'oncoming':oncoming_action,
                                'left':left_action,
                                'right':right_action,
                                'waypoint': waypoint_direction
                            }.items()))
                            self.Q[key] = {}
                            for action in Environment.valid_actions:
                                self.Q[key][action] = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.net_reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        inputs['waypoint'] = self.next_waypoint
        self.state = tuple(sorted(inputs.items()))

        # TODO: Select action according to your policy
        action = random.choice(Environment.valid_actions[1:])
        # Choose to either follow waypoint or stay.
        # Comment following 3 lines to allow agent to move randomly.
        action = self.next_waypoint
        if self.Q[self.state][None] > self.Q[self.state][action]:
            action = None
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 1:
            self.total_penalties += 1
        self.total_moves += 1
        self.net_reward += reward
        # pdb.set_trace()

        # TODO: Learn policy based on state, action, reward
        self.Q[self.state][action] += reward * self.learning_rate

        self.env.status_text += "  penalties/moves = {}/{}  net reward={}".format(
            self.total_penalties,self.total_moves, self.net_reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
