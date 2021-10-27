from tf_agents.agents import tf_agent
import tensorflow as tf
from twoWaySignPolicy import TwoWaySignPolicy

class SignAgent(tf_agent.TFAgent):
  def __init__(self):
    self._situation = tf.Variable(0, dtype=tf.int32)
    policy = TwoWaySignPolicy(self._situation)
    time_step_spec = policy.time_step_spec
    action_spec = policy.action_spec
    super(SignAgent, self).__init__(time_step_spec=time_step_spec,
                                    action_spec=action_spec,
                                    policy=policy,
                                    collect_policy=policy,
                                    train_sequence_length=None)

  def _initialize(self):
    return tf.compat.v1.variables_initializer(self.variables)

  def _train(self, experience, weights=None):
    observation = experience.observation
    action = experience.action
    reward = experience.reward

    # We only need to change the value of the situation variable if it is
    # unknown (0) right now, and we can infer the situation only if the
    # observation is not 0.
    needs_action = tf.logical_and(tf.equal(self._situation, 0),
                                  tf.not_equal(reward, 0))


    def new_situation_fn():
      """This returns either 1 or 2, depending on the signs."""
      return (3 - tf.sign(tf.cast(observation[0, 0, 0], dtype=tf.int32) *
                          tf.cast(action[0, 0], dtype=tf.int32) *
                          tf.cast(reward[0, 0], dtype=tf.int32))) / 2

    new_situation = tf.cond(needs_action,
                            new_situation_fn,
                            lambda: self._situation)
    new_situation = tf.cast(new_situation, tf.int32)
    tf.compat.v1.assign(self._situation, new_situation)
    return tf_agent.LossInfo((), ())
