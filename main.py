""" You need install the following package:
    pip install stable-baselines3
    pip install stable-baselines3[extra]
    pip install tensorboard==2.10.1
"""

import gymnasium as gym
import numpy as np

from rsac import RSAC
from stable_baselines3 import SAC
from rsac_plot import figure_robustness,figure_distribution_pendulum

risk_coef_list = [-0.03,-0.02,-0.01,0.01,0.02,0.03]
ent_coef_list = [0.01,0.1,1.0]

def train_SAC(ent_coef_idx=1,name_idx = 0):
    """train a standard SAC policy"""
    env = gym.make("Pendulum-v1",render_mode='rgb_array')
    loadpath = 'sac_E{}'.format(ent_coef_idx)
    prefix = './data/'
    logpath = prefix+loadpath+'_tensorboard/'
    model =  SAC('MlpPolicy', env, verbose=1,
                ent_coef=ent_coef_list[1],tensorboard_log = logpath,
                buffer_size=int(10e5),batch_size=256,learning_rate=0.001)

    model.learn(50000,log_interval=2,progress_bar=True)
    model.save(prefix+loadpath+'_{}'.format(name_idx))


def train_RSAC(risk_coef_idx=3,ent_coef_idx=1,name_idx = 0):
    """train a risk sensitive SAC policy"""
    env = gym.make("Pendulum-v1",render_mode='rgb_array')
    loadpath = 'sac_R{}_E{}'.format(risk_coef_idx,ent_coef_idx)
    prefix = './data/'
    logpath = prefix+loadpath+'_tensorboard/'
    model =  RSAC('MlpPolicy', env, verbose=1,
                risk_coef=risk_coef_list[risk_coef_idx],
                ent_coef=ent_coef_list[ent_coef_idx],tensorboard_log = logpath,
                buffer_size=int(10e5),batch_size=256,learning_rate=0.001)

    model.learn(50000,log_interval=2,progress_bar=True)
    model.save(prefix+loadpath+'_{}'.format(name_idx))

def evaluate(risk_coef_idx=3,ent_coef_idx=0,name_idx = 0,l = 1.0,n_sample = 100):
    # You need to modify PendulumEnv by
    # def __init__(self, render_mode: Optional[str] = None, g=10.0,l=1.0):
    # ......
    # self.l = l 
    # ......
    # to evaluate the robustness (change the pole length l)
    env = gym.make("Pendulum-v1",render_mode='rgb_array',l=l)
    prefix = './data/'
    if risk_coef_idx is None:
        loadpath = 'sac_E{}'.format(ent_coef_idx)
        model =  SAC('MlpPolicy', env, verbose=1,
                ent_coef=ent_coef_list[ent_coef_idx],
                buffer_size=int(10e5),batch_size=256,learning_rate=0.001)
    else:
        loadpath = 'sac_R{}_E{}'.format(risk_coef_idx,ent_coef_idx)
        model =  RSAC('MlpPolicy', env, verbose=1,
                risk_coef=risk_coef_list[risk_coef_idx],
                ent_coef=ent_coef_list[ent_coef_idx],
                buffer_size=int(10e5),batch_size=256,learning_rate=0.001)

    model= model.load(prefix+loadpath+"_{}".format(name_idx),env)   
    ep_reward_list =[]
    for ep in range(n_sample):
        #obs,_ = env.reset(options={"x_init":0,"y_init":0})
        obs,_ = env.reset()

        ep_reward =0
        for i in range(200):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, _,_, info = env.step(action)
            ep_reward+= reward
        ep_reward_list.append(ep_reward)
        #print(ep,ep_reward)
    ep_reward_mean = np.mean(ep_reward_list)
    print("Average episode reward:",ep_reward_mean)
    return ep_reward_mean

def robustness_data(l_list = [1.0,1.25,1.5], N=10, n_sample=100):
    """generate the robustness testing data"""
    for l in l_list:
        data_set = []
        for i in range(N):
            # SAC robustness data
            sac_eprw = evaluate(None,1,0,l=l,n_sample=n_sample)

            # RSAC robustness data
            rsac_eprw1 = evaluate(1,1,0,l=l,n_sample=n_sample)
            rsac_eprw2 = evaluate(2,1,0,l=l,n_sample=n_sample)
            rsac_eprw3 = evaluate(3,1,0,l=l,n_sample=n_sample)
            rsac_eprw4 = evaluate(4,1,0,l=l,n_sample=n_sample)

            data_set.append([rsac_eprw1,rsac_eprw2,sac_eprw,rsac_eprw3,rsac_eprw4])
            print(l, i)
        data_set = np.array(data_set)
        prefix = './data/'
        np.savetxt(prefix+"robustness_dataset_l{}.csv".format(l), data_set)

if __name__ == '__main__':
    """Step1: train standard SAC"""
    #train_SAC(ent_coef_idx=1)

    """Step2: train RSAC for eta = -0.02,-0.01, 0.01, 0.02"""
    #for i in range(4):
        #train_RSAC(risk_coef_idx=i+1,ent_coef_idx=1)

    """Step3: generate robustness testing data"""
    #robustness_data(N=20,n_sample=100)
    
    """Step4: plot the graph"""
    figure_robustness()
    figure_distribution_pendulum()
    
