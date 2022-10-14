# Model-based Reinforcement Learning for navigation using VAE. For Testing
# 2022.08.02    ANGEL CANELO

######### IMPORT ##########
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Add, Activation, Input, experimental, Concatenate
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import seaborn as sns
###########################
#### Action-Obs spaces ####
tf.keras.backend.clear_session()
num_states = 7
num_actions = 2
upper_bound1 = 1.75
upper_bound2 = 3
##### Actor net ###########
def policy_decoder():
  last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
  act_in = Input(shape=(num_states), name='ActorIn')
  lat_in = Input(shape=(2,))
  full_in = Concatenate()([act_in, lat_in])
  act = Dense(400, activation='relu', kernel_initializer='glorot_uniform', name='ActorL1')(full_in)
  act = Dense(300, activation='relu', kernel_initializer='glorot_uniform', name='ActorL2')(act)
  act_out1 = Dense(1, activation='tanh', kernel_initializer=last_init, bias_initializer=last_init, name='ActorL3')(act)
  act_out2 = Dense(1, activation='tanh', kernel_initializer=last_init, bias_initializer=last_init, name='ActorL4')(act)
  act_out1 = act_out1 * upper_bound1
  act_out2 = act_out2 * upper_bound2
  act_out = Concatenate()([act_out1, act_out2])
  pol_dec = Model([act_in, lat_in], act_out, name='Policy_decoder')
  return pol_dec

def encoder():
  inp = Input(shape=(num_states,), name='EncoderIn')
  enco = Dense(400, activation='relu', kernel_initializer='glorot_uniform', name='EncoderL1')(inp)
  enco = Dense(300, activation='relu', kernel_initializer='glorot_uniform')(enco)
  mu = Dense(1, activation='linear')(enco)
  log_sigma = Dense(1, activation='linear')(enco)
  mu2 = Dense(1, activation='linear')(enco)
  log_sigma2 = Dense(1, activation='linear')(enco)
  enc1 = mu + tf.math.exp(log_sigma) * tf.random.normal(shape=(1,), mean=0, stddev=0.1)
  enc2 = mu2 + tf.math.exp(log_sigma2) * tf.random.normal(shape=(1,), mean=0, stddev=0.1)
  enc = Concatenate()([enc1, enc2, mu, log_sigma, mu2, log_sigma2])
  encm = Model(inputs=inp, outputs=enc, name='Encoder')
  return encm

def encoder_circ():
  inp = Input(shape=(2,), name='EncoderIn')
  inp2 = Input(shape=(2,))
  encc = Concatenate()([inp,inp2])
  encc = Dense(100, activation='relu', kernel_initializer='glorot_uniform', name='EncoderL1')(encc)
  encc = Dense(100, activation='relu', kernel_initializer='glorot_uniform')(encc)
  encc = Dense(2, activation='linear')(encc)
  encc = Model(inputs=[inp, inp2], outputs=encc, name='Encoder')
  return encc

def decoder():
  inp = Input(shape=(2,))
  dec = Dense(400, activation='relu', kernel_initializer='glorot_uniform', name='DecoderL1')(inp)
  dec = Dense(300, activation='relu', kernel_initializer='glorot_uniform')(dec)
  dec = Dense(num_states, activation='linear')(dec)
  decod = Model(inputs=inp, outputs=dec, name='State_decoder')
  return decod
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
#### Choose best state ####
def choose_state(lat, ss):
  z_mean = np.squeeze(lat[:, 2:5:2])
  z_logsigma = np.squeeze(lat[:, 3:6:2])
  dist = tfd.Normal(loc=z_mean, scale=tf.math.exp(z_logsigma) * tf.random.normal(shape=(2,), mean=0, stddev=0.1))
  lat_sequence = dist.sample([20])
  ss_rep = tf.repeat(ss, repeats=20, axis=0)
  lat_sequence2 = encoder_circ_model([lat_sequence, ss_rep])
  predic_state = decoder_model(lat_sequence2)
  pred_reward = -((predic_state[:, 2] ** 2) * 0.1 + (predic_state[:, 3] ** 2) * 0.1 +
                  (predic_state[:, 0] ** 2) * 0.1 + (predic_state[:, 5] ** 2) * 0.01 +
                  (predic_state[:, 6] ** 2) * 0.01)
  best_zind = np.argmax(pred_reward)
  best_z = tf.expand_dims(lat_sequence2[best_zind, :], axis=0)
  return best_z
###########################
actor_model = policy_decoder()
encoder_model = encoder()
encoder_circ_model = encoder_circ()
decoder_model = decoder()
###########################
actor_model.load_weights("../weights/policy_{}.h5".format(1))
encoder_model.load_weights("../weights/encoder_{}.h5".format(1))
encoder_circ_model.load_weights("../weights/encoder_compass_{}.h5".format(1))
decoder_model.load_weights("../weights/decoder_{}.h5".format(1))
###########################
####### VAE-DDPG #####
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

y1 = np.zeros(t0.shape[0]); y2 = np.zeros(t0.shape[0]); y3 = np.zeros(t0.shape[0])
phi = 0
lat_space_x = []; lat_space_y = []; state_label_Tor = []
for jj in range(t0.shape[0] - 1):
  state = tf.expand_dims(tf.convert_to_tensor(np.array([phi, y1[jj], X[jj]-xy1[jj], Y[jj]-yy1[jj], y2[jj], xy2[jj], yy2[jj]])),0)
  latent = encoder_model(state)
  ss = tf.expand_dims(tf.stack([X[jj]-xy1[jj], Y[jj]-yy1[jj]]), axis=0)
  lat = choose_state(latent,ss)

  lat_space_x.append(lat[:, 0])
  lat_space_y.append(lat[:, 1])

  a1, a2 = tf.squeeze(actor_model([state, lat]))

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
  state_label_Tor.append(np.linalg.norm(Pv))
###########################
actor_model = get_actor()
actor_model.load_weights("../weights/actor_DDPG_{}.h5".format(1))
###########################
####### DDPG #####
Bxy1 = np.zeros(t0.shape[0]); Bxy2 = np.zeros(t0.shape[0]); Bxy3 = np.zeros(t0.shape[0])
Byy1 = np.zeros(t0.shape[0]); Byy2 = np.zeros(t0.shape[0]); Byy3 = np.zeros(t0.shape[0])

By1 = np.zeros(t0.shape[0]); By2 = np.zeros(t0.shape[0]); By3 = np.zeros(t0.shape[0])
phi = 0
for jj in range(t0.shape[0] - 1):

  a1, a2 = np.squeeze(actor_model.predict(np.expand_dims(np.array([phi, By1[jj], X[jj]-Bxy1[jj], Y[jj]-Byy1[jj],
                                                                   By2[jj], Bxy2[jj], Byy2[jj]]),0)))
  By3[jj] = a1
  Bxy3[jj] = a2*np.sin(By1[jj])
  Byy3[jj] = a2*np.cos(By1[jj])

  By1[jj + 1] = By1[jj] + dt * By2[jj]
  By2[jj + 1] = By2[jj] + dt / I * (-betaz * By2[jj] + By3[jj])

  Bxy1[jj + 1] = Bxy1[jj] + dt * Bxy2[jj]
  Bxy2[jj + 1] = Bxy2[jj] + dt * (-beta * Bxy2[jj] + Bxy3[jj]) / Mass

  Byy1[jj + 1] = Byy1[jj] + dt * Byy2[jj]
  Byy2[jj + 1] = Byy2[jj] + dt * (-beta * Byy2[jj] + Byy3[jj]) / Mass

  Pv = [X[jj]-Bxy1[jj+1], Y[jj]-Byy1[jj+1]]
  Vv = [Bxy2[jj + 1], Byy2[jj + 1]]
  unPv = Pv / np.linalg.norm(Pv)
  unVv = Vv / np.linalg.norm(Vv)
  dotprod = np.dot(unPv, unVv)
  phi = np.arccos(dotprod)
###########################
actor_model = get_actor()
actor_model.load_weights("../weights/actor_TD3_{}.h5".format(1))
####### TD3 #####
Cxy1 = np.zeros(t0.shape[0]); Cxy2 = np.zeros(t0.shape[0]); Cxy3 = np.zeros(t0.shape[0])
Cyy1 = np.zeros(t0.shape[0]); Cyy2 = np.zeros(t0.shape[0]); Cyy3 = np.zeros(t0.shape[0])

Cy1 = np.zeros(t0.shape[0]); Cy2 = np.zeros(t0.shape[0]); Cy3 = np.zeros(t0.shape[0])
phi = 0
for jj in range(t0.shape[0] - 1):

  a1, a2 = np.squeeze(actor_model.predict(np.expand_dims(np.array([phi, Cy1[jj], X[jj]-Cxy1[jj], Y[jj]-Cyy1[jj],
                                                                   Cy2[jj], Cxy2[jj], Cyy2[jj]]),0)))
  Cy3[jj] = a1
  Cxy3[jj] = a2*np.sin(Cy1[jj])
  Cyy3[jj] = a2*np.cos(Cy1[jj])

  Cy1[jj + 1] = Cy1[jj] + dt * Cy2[jj]
  Cy2[jj + 1] = Cy2[jj] + dt / I * (-betaz * Cy2[jj] + Cy3[jj])

  Cxy1[jj + 1] = Cxy1[jj] + dt * Cxy2[jj]
  Cxy2[jj + 1] = Cxy2[jj] + dt * (-beta * Cxy2[jj] + Cxy3[jj]) / Mass

  Cyy1[jj + 1] = Cyy1[jj] + dt * Cyy2[jj]
  Cyy2[jj + 1] = Cyy2[jj] + dt * (-beta * Cyy2[jj] + Cyy3[jj]) / Mass

  Pv = [X[jj]-Cxy1[jj+1], Y[jj]-Cyy1[jj+1]]
  Vv = [Cxy2[jj + 1], Cyy2[jj + 1]]
  unPv = Pv / np.linalg.norm(Pv)
  unVv = Vv / np.linalg.norm(Vv)
  dotprod = np.dot(unPv, unVv)
  phi = np.arccos(dotprod)
##########################################################################
##### Save and plot #######
# Plotting training performance
sns.set()
# Plotting test environment simulation
fig2 = plt.figure(figsize=(10,6))
ax2 = fig2.subplots(nrows=1, ncols=2)
line, = ax2[0].plot(xy1, yy1, linewidth=1, color='blue')
line2, = ax2[0].plot(Cxy1, Cyy1, linewidth=1, color='green')
line3, = ax2[0].plot(Bxy1, Byy1, linewidth=1, color='red')
ax2[0].plot(X, Y, marker='*', linewidth=1, color='black')
ax2[0].legend(['Ours', 'TD3', 'DDPG', 'Target'], loc='upper left', frameon=False)
ax2[0].set_title("Navigation environment")
ax2[0].set_xlabel('X Position (m)'); ax2[0].set_ylabel('Y Position (m)')

# Plotting Latent space

alpha = np.arange(-np.pi,np.pi,0.01)
scatt = ax2[1].scatter(lat_space_x, lat_space_y, c=state_label_Tor, cmap='jet', alpha=0.5)
cbar = fig2.colorbar(scatt)
cbar.set_label(label='Euclidean distance (m)')
ax2[1].set_title("Latent space\n(Relative position with respect to target)")
ax2[1].set_xlabel("z'1"); ax2[1].set_ylabel("z'2")
ax2[1].set_ylim([-1.5,1.5]); ax2[1].set_xlim([-1.5,1.5])
ax2[0].set_aspect(1)
ax2[1].set_aspect(1)

plt.show()