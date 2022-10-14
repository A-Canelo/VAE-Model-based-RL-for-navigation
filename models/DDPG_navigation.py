# navigation using DDPG. For Training
# 2022.08.02    ANGEL CANELO

######### IMPORT ##########
from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import random
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Add, Activation, Input, experimental, Concatenate
from tensorflow.keras.regularizers import l1, l2
from scipy.io import savemat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
###########################
###### ENVIRONMENT ########
class Maze(Env):
  def __init__(self):
    # Action Force. Indicate max and min value that the force can take.
    self.action_space = Box(low=np.array([-1.75, -3]), high=np.array([1.75, 3]), dtype=np.float32)    # X,Y 2D force
    # Observations. This will be the input to the neural network. Error distance between target and fly. Xe,Ye
    self.observation_space = Box(low=np.array([-np.pi,-np.pi, -4, -4, -math.inf, -math.inf, -math.inf]),
                                 high=np.array([np.pi,np.pi, 4, 4, math.inf, math.inf, math.inf]), dtype=np.float32)
    self.min_ = -3
    self.max_ = 3
    self.min = -1.75
    self.max = 1.75

    self.t = 0
    self.dt = 0.01
    self.tmax = 2
    self.seed()
    self.betaz = 1e-1
    self.I = 6e-4
    self.beta = 1e-5 # drag coefficient
    self.Mass = 10 / 9.81 # Weight of the fly is 10uN https: // bionumbers.hms.harvard.edu / bionumber.aspx?id = 110030 & ver = 2
    self.rnx = self.np_random.uniform(low=-2, high=2)
    self.rny = self.np_random.uniform(low=-2, high=2)
    self.X = self.rnx         # Target position
    self.Xprime = 0          # Target velocity
    self.xy1 = 0             # Fly position
    self.xy2 = 0             # Fly velocity
    self.Y = self.rny        # Target position
    self.Yprime = 0          # Target velocity
    self.yy1 = 0             # Fly position
    self.yy2 = 0             # Fly velocity
    self.psi_obj = np.arctan(self.X/self.Y)
    self.psi_obj_dot = 0
    self.yode1 = 0; self.yode2 = 0
    self.phi = 0
    self.state = np.array([self.phi, self.yode1, self.X - self.xy1, self.Y - self.yy1, self.yode2, self.xy2, self.yy2])

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    # Apply action. Forces comes from the neural network
    Tor, F = action
    Tor = np.clip(Tor, self.min, self.max)  # Torque
    F = np.clip(F, self.min_, self.max_)  # Force
    # Calculate reward
    reward = -(((self.X - self.xy1) ** 2) * 0.1 + ((self.Y - self.yy1) ** 2) * 0.1 +
               ((self.phi) ** 2) * 0.1 + #((self.psi_obj_dot - self.yode2) ** 2) * 0.01
                ((self.Xprime-self.xy2)**2)*0.01 + ((self.Yprime-self.yy2)**2)*0.01)
    if 180/np.pi*abs(self.phi) < 2:
      reward = reward + 1
    if abs(self.X - self.xy1) < 0.1 and abs(self.Y - self.yy1) < 0.1:
      reward = reward + 10

    # Steering control psi
    self.yode1 = self.yode1 + self.dt * self.yode2
    self.yode2 = self.yode2 + self.dt / self.I * (-self.betaz * self.yode2 + Tor)
    ''' Euler method. Fx and Fy substitute the equations xy3 and yy3 respectively (from Matlab)'''
    # X coordinate control (Translational force)
    self.xy1 = self.xy1 + self.dt*self.xy2
    self.xy2 = self.xy2 + self.dt/self.Mass*(-self.beta*self.xy2+F*np.sin(self.yode1))
    # Y coordinate control (Translational force)
    self.yy1 = self.yy1 + self.dt*self.yy2
    self.yy2 = self.yy2 + self.dt/self.Mass*(-self.beta*self.yy2+F*np.cos(self.yode1))

    self.Pv = [self.X - self.xy1, self.Y - self.yy1]
    self.Vv = [self.xy2, self.yy2]
    unPv = self.Pv / np.linalg.norm(self.Pv)
    unVv = self.Vv / np.linalg.norm(self.Vv)
    dotprod = np.dot(unPv, unVv)
    self.phi = np.arccos(dotprod)

    self.state = np.array([self.phi, self.yode1, self.X - self.xy1, self.Y - self.yy1, self.yode2, self.xy2, self.yy2])

    self.t += self.dt
      # Check if done
    if self.t >= self.tmax:
      done = True
    else:
      done = False

    # Set placeholder for info
    info = {}

    # Return step information
    return self.state, reward, done, info, self.xy1, self.yy1

  def reset(self):    # At the beginning of each episode we reset variables
    self.t = 0
    self.rnx = self.np_random.uniform(low=-2, high=2)
    self.rny = self.np_random.uniform(low=-2, high=2)
    self.X= self.rnx         # Target position
    self.Xprime = 0          # Target velocity
    self.xy1 = 0             # Fly position
    self.xy2 = 0             # Fly velocity
    self.Y = self.rny        # Target position
    self.Yprime = 0          # Target velocity
    self.yy1 = 0             # Fly position
    self.yy2 = 0             # Fly velocity
    self.psi_obj = np.arctan(self.X/self.Y)
    self.psi_obj_dot = 0
    self.yode1 = 0; self.yode2 = 0
    self.phi = 0
    self.state = np.array([self.phi, self.yode1, self.X - self.xy1, self.Y - self.yy1, self.yode2, self.xy2, self.yy2])
    return self.state
###########################
##Ornstein-Uhlenbeck_process##
class OUActionNoise:
  def __init__(self, mean, std_deviation, theta=0.15, dt=0.05, x_initial=None):
    self.theta = theta
    self.mean = mean
    self.std_dev = std_deviation
    self.dt = dt
    self.x_initial = x_initial
    self.reset()

  def __call__(self):
    # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
    )
    # Store x into x_prev
    # Makes next noise dependent on current one
    self.x_prev = x
    return x

  def reset(self):
    if self.x_initial is not None:
      self.x_prev = self.x_initial
    else:
      self.x_prev = np.zeros_like(self.mean)
###########################
####### BUFFER ############
class Buffer:
  def __init__(self, buffer_capacity=100000, batch_size=64):
    # Number of "experiences" to store at max
    self.buffer_capacity = buffer_capacity
    # Num of tuples to train on.
    self.batch_size = batch_size

    # Its tells us num of times record() was called.
    self.buffer_counter = 0

    # Instead of list of tuples as the exp.replay concept go
    # We use different np.arrays for each tuple element
    self.state_buffer = np.zeros((self.buffer_capacity, num_states))
    self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
    self.reward_buffer = np.zeros((self.buffer_capacity, 1))
    self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

  # Takes (s,a,r,s') obervation tuple as input
  def record(self, obs_tuple):
    # Set index to zero if buffer_capacity is exceeded,
    # replacing old records
    index = self.buffer_counter % self.buffer_capacity

    self.state_buffer[index] = obs_tuple[0]
    self.action_buffer[index] = obs_tuple[1]
    self.reward_buffer[index] = obs_tuple[2]
    self.next_state_buffer[index] = obs_tuple[3]

    self.buffer_counter += 1

  # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
  # TensorFlow to build a static graph out of the logic and computations in our function.
  # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
  @tf.function
  def update(
          self, state_batch, action_batch, reward_batch, next_state_batch,
  ):
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.
    with tf.GradientTape() as tape:
      target_actions = target_actor(next_state_batch, training=True)
      y = reward_batch + gamma * target_critic(
        [next_state_batch, target_actions], training=True
      )
      critic_value = critic_model([state_batch, action_batch], training=True)
      critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
      zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
      actions = actor_model(state_batch, training=True)
      critic_value = critic_model([state_batch, actions], training=True)
      # Used `-value` as we want to maximize the value given
      # by the critic for our actions
      actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
      zip(actor_grad, actor_model.trainable_variables)
    )

  # We compute the loss and update parameters
  def learn(self):
    # Get sampling range
    record_range = min(self.buffer_counter, self.buffer_capacity)
    # Randomly sample indices
    batch_indices = np.random.choice(record_range, self.batch_size)

    # Convert to tensors
    state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
    action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
    reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
    reward_batch = tf.cast(reward_batch, dtype=tf.float32)
    next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

    self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
  for (a, b) in zip(target_weights, weights):
    a.assign(b * tau + a * (1 - tau))
###########################
#### Action-Obs spaces ####
tf.keras.backend.clear_session()
env = Maze()
num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound1 = env.action_space.high[0]
lower_bound1 = env.action_space.low[0]
upper_bound2 = env.action_space.high[1]
lower_bound2 = env.action_space.low[1]
print("Max Value of Action ->  {}".format(upper_bound1))
print("Min Value of Action ->  {}".format(lower_bound1))
###########################
##### Critic net ##########
def get_critic():
  last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
  observation_input = Input(shape=(num_states), name='observation_input')
  action_input = Input(shape=(num_actions), name='action_input')
  crit = Concatenate(name='CriticL3')([observation_input,action_input])
  crit = Dense(400, activation='relu', kernel_initializer='glorot_uniform', name='CriticL4')(crit)
  crit = Dense(300, activation='relu', kernel_initializer='glorot_uniform', name='CriticL5')(crit)
  crit_out = Dense(1, kernel_initializer=last_init, bias_initializer=last_init, name='CriticL6')(crit)
  critModel = Model(inputs=[observation_input, action_input], outputs=crit_out, name='Critic')
  return critModel
###########################
##### Actor net ###########
def get_actor():
  last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
  act_in = Input(shape=(num_states,), name='ActorIn')
  act = Dense(400, activation='relu', kernel_initializer='glorot_uniform', name='ActorL1')(act_in)
  act = Dense(300, activation='relu', kernel_initializer='glorot_uniform', name='ActorL2')(act)
  act_out1 = Dense(1, activation='tanh', kernel_initializer=last_init, bias_initializer=last_init, name='ActorL3')(act)
  act_out2 = Dense(1, activation='tanh', kernel_initializer=last_init, bias_initializer=last_init, name='ActorL4')(act)
  act_out1 = act_out1 * upper_bound1
  act_out2 = act_out2 * upper_bound2
  act_out = Concatenate()([act_out1, act_out2])
  actModel = Model(act_in, act_out, name='Actor')
  return actModel
###########################
## Policy (sampling net) ##
def policy(state, noise_object):
  sampled_actions = tf.squeeze(actor_model(state))
  noise = noise_object()
  # Adding noise to action
  sampled_actions = np.squeeze(sampled_actions) + noise

  return np.squeeze(sampled_actions)
###########################
##### Hyperparameters #####
ou_noise = OUActionNoise(mean=np.array([0,0]), std_deviation=np.array([0.3,0.3]))
to_file = {"Trial": [], "Reward": [], "Dim1": [], "Dim2":[], "Label1":[], "PosX":[], "PosY":[],
                  "VelX":[], "VelY":[],"Algorithm": []}

for kk in range(2):
  print('Trial', kk + 1, 'of', 2)
  tf.keras.backend.clear_session()
  actor_model = get_actor()
  critic_model = get_critic()

  target_actor = get_actor()  # Target networks learn slowly to maintain stability
  target_critic = get_critic()

  # Making the weights equal initially
  target_actor.set_weights(actor_model.get_weights())
  target_critic.set_weights(critic_model.get_weights())

  # Learning rate for actor-critic models
  # critic_lr = 0.002
  # actor_lr = 0.001
  critic_lr = 1e-3
  actor_lr = 1e-4

  critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
  actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

  # Discount factor for future rewards
  gamma = 0.99
  # Used to update target networks
  tau = 0.001  # 0.005

  buffer = Buffer(50000, 256)  # buffer = Buffer(50000, 64)
  ###########################
  ##### Training ############
  # To store reward history of each episode
  ep_reward_list = []
  # To store average reward history of last few episodes
  avg_reward_list = []
  # Takes about 4 min to train
  total_episodes = 2000
  avg_reward = 0
  for i in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:

      tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

      action = policy(tf_prev_state, ou_noise)
      # Recieve state and reward from environment.
      state, reward, done, info, psi_fly, psi_fly_dot = env.step(action)

      buffer.record((prev_state, action, reward, state))
      episodic_reward += reward

      buffer.learn()
      update_target(target_actor.variables, actor_model.variables, tau)
      update_target(target_critic.variables, critic_model.variables, tau)

      # End this episode when `done` is True
      if done:
        break

      prev_state = state
    ep_reward_list.append(episodic_reward)

    # Mean of last 50 episodes
    avg_reward = np.mean(ep_reward_list[-50:])
    avg_reward_list.append(avg_reward)
    print("Episode * {} * Ep Reward is ==> {} * Avg Reward is ==> {}".format(i + 1, round(episodic_reward, 2),
                                                                                          round(avg_reward, 2)))
  actor_model.save_weights("fly_actor_DDPG_{}.h5".format(1))
  critic_model.save_weights("fly_critic_DDPG_{}.h5".format(1))
  ###########################
  ####### Euler ODE Bar #####
  dt = 0.01; t0 = np.arange(0, 5 + dt, dt)
  beta = 1e-5; Mass = 10 / 9.81
  betaz = 1e-1; I = 6e-4
  xy1 = np.zeros(t0.shape[0]); xy2 = np.zeros(t0.shape[0]); xy3 = np.zeros(t0.shape[0])
  yy1 = np.zeros(t0.shape[0]); yy2 = np.zeros(t0.shape[0]); yy3 = np.zeros(t0.shape[0])
  X = 1*np.ones(t0.shape[0])
  Y = 1*np.ones(t0.shape[0])
  X_copy = np.zeros(t0.shape[0] + 1); X_copy[0] = X[0]
  X_copy[1:len(X_copy)] = X
  Xprime_bar = np.diff(X_copy) / (t0[1] - t0[0])
  Y_copy = np.zeros(t0.shape[0] + 1); Y_copy[0] = Y[0]
  Y_copy[1:len(Y_copy)] = Y
  Yprime_bar = np.diff(Y_copy) / (t0[1] - t0[0])

  theta_bar = np.arctan(X/Y)
  theta_bar_copy = np.zeros(t0.shape[0] + 1); theta_bar_copy[0] = theta_bar[0]
  theta_bar_copy[1:len(theta_bar_copy)] = theta_bar
  dot_theta_bar = np.diff(theta_bar_copy) / (t0[1] - t0[0])
  y1 = np.zeros(t0.shape[0]); y2 = np.zeros(t0.shape[0]); y3 = np.zeros(t0.shape[0])
  phi = 0
  for jj in range(t0.shape[0] - 1):

    a1, a2 = np.squeeze(actor_model.predict(np.expand_dims(np.array([phi, y1[jj], X[jj]-xy1[jj], Y[jj]-yy1[jj],
                                                                     y2[jj], xy2[jj], yy2[jj]]),0)))
    y3[jj] = a1
    xy3[jj] = a2*np.sin(y1[jj])
    yy3[jj] = a2*np.cos(y1[jj])

    y1[jj + 1] = y1[jj] + dt * y2[jj]
    y2[jj + 1] = y2[jj] + dt / I * (-betaz * y2[jj] + y3[jj])

    xy1[jj + 1] = xy1[jj] + dt * xy2[jj]
    xy2[jj + 1] = xy2[jj] + dt * (-beta * xy2[jj] + xy3[jj]) / Mass

    yy1[jj + 1] = yy1[jj] + dt * yy2[jj]
    yy2[jj + 1] = yy2[jj] + dt * (-beta * yy2[jj] + yy3[jj]) / Mass

    Pv = [X[jj]-xy1[jj+1], Y[jj]-yy1[jj+1]]
    Vv = [xy2[jj + 1], yy2[jj + 1]]
    unPv = Pv / np.linalg.norm(Pv)
    unVv = Vv / np.linalg.norm(Vv)
    dotprod = np.dot(unPv, unVv)
    phi = np.arccos(dotprod)
  ###########################
  ##### Save and plot #######
  to_file["Trial"].append(kk)
  to_file["Reward"].append(avg_reward_list)
  to_file["PosX"].append(xy1)
  to_file["PosY"].append(yy1)
  to_file["VelX"].append(xy2)
  to_file["VelY"].append(yy2)
  to_file["Algorithm"].append("DDPG_vanilla")

savemat("Reward_DDPGvanilla_01.mat", to_file)
rew_dic = {"Reward": [], "Episode": [], "Trial": [], "Algorithm": []}
for j in range(len(to_file["Trial"])):
  for i in range(len(to_file["Reward"][0])):
    rew_dic["Reward"].append(to_file["Reward"][j][i])
    rew_dic["Episode"].append(i)
    rew_dic["Trial"].append(j)
    rew_dic["Algorithm"].append("DDPG_vanilla")
df_r = pd.DataFrame(data=rew_dic, columns=["Episode", "Reward", "Trial", "Algorithm"])
sns.set()
plt.figure();
plt.clf()
sns.lineplot(x="Episode", y="Reward", data=df_r, hue="Algorithm", ci=95, palette=['blue'])
plt.ylabel('Average reward');
plt.xlabel('Episode')
plt.legend(loc='upper left')
plt.show()