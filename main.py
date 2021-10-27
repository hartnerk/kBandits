import abc
import numpy as np
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step


#Broken out classes
from banditPyEnvironment import BanditPyEnvironment
from simplePyEnvironment import SimplePyEnvironment
from twoWayPyEnvironment import TwoWayPyEnvironment
from signAgent import SignAgent
from twoWaySignPolicy import TwoWaySignPolicy
from signPolicy import SignPolicy
    
nest = tf.nest

two_way_tf_environment = tf_py_environment.TFPyEnvironment(TwoWayPyEnvironment())

sign_agent = SignAgent()


environment = SimplePyEnvironment()
observation = environment.reset().observation
print("observation: %d" % observation)

action = 2 #@param

print("action: %d" % action)
reward = environment.step(action).reward
print("reward: %f" % reward)

tf_environment = tf_py_environment.TFPyEnvironment(environment)

sign_policy = SignPolicy()

current_time_step = tf_environment.reset()
print('Observation:')
print (current_time_step.observation)
action = sign_policy.action(current_time_step).action
print('Action:')
print (action)
reward = tf_environment.step(action).reward
print('Reward:')
print(reward)

step = tf_environment.reset()
action = 1
next_step = tf_environment.step(action)
reward = next_step.reward
next_observation = next_step.observation
print("Reward: ")
print(reward)
print("Next observation:")
print(next_observation)

# We need to add another dimension here because the agent expects the
# trajectory of shape [batch_size, time, ...], but in this tutorial we assume
# that both batch size and time are 1. Hence all the expand_dims.

def trajectory_for_bandit(initial_step, action_step, final_step):
  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))

step = two_way_tf_environment.reset()

for _ in range(10):
  action_step = sign_agent.collect_policy.action(step)
  next_step = two_way_tf_environment.step(action_step.action)
  experience = trajectory_for_bandit(step, action_step, next_step)
  print(experience)
  sign_agent.train(experience)
  step = next_step
