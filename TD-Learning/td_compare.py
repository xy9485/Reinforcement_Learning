import os 
import numpy as np
import random


class Env():                   
    '''构造一个环境类'''        
    def __init__(self, mu, sigma, nB, random_reward=True):  
        self.mu = mu
        self.sigma = sigma
        self.STATE_A = self.left = 0
        self.STATE_B = self.right = 1
        self.Terminal = 2
        self.nS = 3   # 加上Terminal即3个状态
        self.nA = 2
        self.nB = nB  # 状态B的动作数
        self.state = self.STATE_A 
        self.random_reward = random_reward
        
    def reset(self):
        self.state = self.STATE_A
        return self.state
        
    def step(self, action):
        # A--left
        if self.state == self.STATE_A and action == self.left:
            self.state = self.STATE_B
            return self.state, 0, False  # next_state, reward, done
        # A--right
        elif self.state == self.STATE_A and action == self.right:
            self.state = self.Terminal
            return self.state, 0, True
        # B--all_actions
        elif self.state == self.STATE_B:
            self.state = self.Terminal
            # reward = random.normalvariate(self.mu, self.sigma)
            if self.random_reward:
                reward = np.random.normal(self.mu, self.sigma)
            else:
                reward = self.mu

            return self.state, reward, True 

def init_V_table(env):
    '''初始化V表'''
    V = {env.STATE_A:0, env.STATE_B:0, env.Terminal:0}

    # from collections import UserDict
    # class V_table(UserDict):
    #     def __getitem__(self, key):
    #         value = super().__getitem__(key)
    #         # add a Gaussian noise to the value
    #         return np.random.normal(value, 0.1)
    # V = V_table(V)
    
    return V

def init_Q_table(env):
    '''初始化Q表'''
    Q = {env.STATE_A:{action:0 for action in range(env.nA)},
         env.STATE_B:{action:0 for action in range(env.nB)},
         env.Terminal:{action:0 for action in range(env.nA)}}
    
    # from collections import UserDict
    # class Q_table(UserDict):
    #     def __getitem__(self, key):
    #         value = super().__getitem__(key)
    #         # add a Gaussian noise to the value
    #         return np.random.normal(value, 0.1) 
    # Q = {env.STATE_A:Q_table({action:0 for action in range(env.nA)}),
    #      env.STATE_B:Q_table({action:0 for action in range(env.nB)}),
    #      env.Terminal:Q_table({action:0 for action in range(env.nA)})}        

    return Q

def select_action_behavior_policy(action_value_dict, epsilon):
    '''使用epsilon-greedy采样action'''
    if random.random() > epsilon:   
        max_keys = [key for key, value in action_value_dict.items() if value == max( action_value_dict.values() )]
        action = random.choice(max_keys)
    else:  
        # 从Q字典对应state中随机选取1个动作,由于返回list,因此通过[0]获取元素
        action = random.sample(action_value_dict.keys(), 1)[0]
    return action


def TD_learning(env, method='Q-Learning', alpha_scope=[0.1, 0.01, 0.99], epsilon_scope=[0.2,0.001,0.99], num_of_episode=1000, gamma=0.9, value_noise=0.0):
    '''
    TD学习算法,返回Q表和估计的最优策略
    其中epsilon_scope由高到低衰减,从左到右分别是[最高值,最低值,衰减因子]
    '''
    epsilon = epsilon_scope[0]
    alpha = alpha_scope[0]
    # 1. 初始化Q1表和Q2表
    Q = init_Q_table(env)
    V = init_V_table(env)
    if method == 'Double-Q':
        Q2 = init_Q_table(env)
    bool_A_left = np.zeros(num_of_episode)
    Aleft_Q_values = []
    Aright_Q_values = []
    B_max_Q_values = []
    B_mean_Q_values = []
    A_V_values = []
    B_V_values = []
    episode_rewards = []
    alpha_values = []
    epsilon_values = []
    for num in range(num_of_episode):
        state = env.reset()  # Init S
        episode_reward = 0
        
        while True:
            # 2.通过behavior policy采样action
            if method == 'Double-Q':
                add_Q1_Q2_state = {action: Q1_value + Q2[state][action] for action, Q1_value in Q[state].items()}
                action = select_action_behavior_policy(add_Q1_Q2_state, epsilon)
            else: action = select_action_behavior_policy(Q[state], epsilon)
            if state == env.STATE_A and action == env.left:  
                bool_A_left[int(num)] += 1
            # 3.执行action并观察R和next state
            next_state, reward, done = env.step(action)
            episode_reward += reward
            # 4.更新Q(S,A),使用max操作更新
            if method == 'Q-Learning':
                V[state] += alpha * (reward + gamma*V[next_state] - V[state])
                Q[state][action] += alpha * (reward + gamma*max( Q[next_state].values() ) - Q[state][action])
            elif method == "VQ-Learning":
                # factor = init_factor * (1-num/num_of_episode)
                V[state] += alpha * (reward + gamma*V[next_state] - V[state])
                error = (1-factor)*(reward + gamma*max(Q[next_state].values()) - Q[state][action]) + factor*(reward + gamma*V[next_state]- Q[state][action])
                Q[state][action] += alpha * error
                # Q[state][action] += alpha * (reward + gamma*max( Q[next_state].values() ) - Q[state][action])
                # Q[state][action] += alpha * (reward + gamma*V[next_state]- Q[state][action]) * factor
                # Q[state][action] += alpha * ((reward + gamma*max( Q[next_state].values() ) - Q[state][action]) + (reward + gamma*V[next_state]- Q[state][action]))
            elif method == 'Sarsa':
                V[state] += alpha * (reward + gamma*V[next_state] - V[state])
                action_prime = select_action_behavior_policy(Q[next_state], epsilon)
                Q[state][action] += alpha * (reward + gamma*Q[next_state][action_prime] - Q[state][action])
            elif method == 'Expected_Sarsa':
                V[state] += alpha * (reward + gamma*V[next_state] - V[state])
                Q[state][action] += alpha * (reward + gamma*sum( Q[next_state].values() ) / len(Q[next_state]) - Q[state][action])  
            elif method == 'Action_Distribution':
                V[state] += alpha * (reward + gamma*V[next_state] - V[state])
                Q[state][action] += alpha * (reward + gamma*random.choice(list( Q[next_state].values() )) - Q[state][action])
            elif method == 'Double-Q':
                V[state] += alpha * (reward + gamma*V[next_state] - V[state])
                # Q[state][action] += alpha * (reward + gamma*V[next_state]- Q[state][action])
                # Q2[state][action] += alpha * (reward + gamma*V[next_state]- Q2[state][action]) * 1000

                if random.random() >= 0.5:
                    # 从Q1表中的下一步state找出状态价值最高对应的action视为Q1[state]的最优动作
                    A1 = random.choice( [action for action, value in Q[next_state].items() if value == max( Q[next_state].values() )] )
                    # 将Q1[state]得到的最优动作A1代入到Q2[state][A1]中的值作为Q1[state]的更新
                    Q[state][action] += alpha * (reward + gamma*Q2[next_state][A1] - Q[state][action])
                else:
                    A2 = random.choice( [action for action, value in Q2[next_state].items() if value == max( Q2[next_state].values() )] )
                    Q2[state][action] += alpha * (reward + gamma*Q[next_state][A2] - Q2[state][action])
            print("num:", num, "reward:", round(reward, 5), "V[B]:", round(V[env.STATE_B], 5), "alpha:", alpha)
            if done: break
            state = next_state
            
        Aleft_Q_values.append(Q[env.STATE_A][env.left])
        Aright_Q_values.append(Q[env.STATE_A][env.right])
        B_max_Q_values.append(max(Q[env.STATE_B].values()))
        B_mean_Q_values.append(sum(Q[env.STATE_B].values()) / len(Q[env.STATE_B]))
        A_V_values.append(V[env.STATE_A])
        B_V_values.append(V[env.STATE_B])
        episode_rewards.append(episode_reward)
        alpha_values.append(alpha)
        epsilon_values.append(epsilon)
        # 对epsilon进行衰减
        if epsilon >= epsilon_scope[1]: epsilon *= epsilon_scope[2]
        if alpha >= alpha_scope[1]: alpha *= alpha_scope[2]
        if value_noise>0.0:
            for state, actions in Q.items():
                for action, value in actions.items():
                    Q[state][action] = np.random.normal(Q[state][action], value_noise)
                    if method == 'Double-Q':
                        Q2[state][action] = np.random.normal(Q2[state][action], value_noise)
            for state, value in V.items():
                V[state] = np.random.normal(V[state], value_noise)
        # if num % 20 == 0:  print("Episode: {}, Score: {}".format(num, sum_reward))
    # convert into numpy array
    Aleft_Q_values = np.array(Aleft_Q_values)
    Aright_Q_values = np.array(Aright_Q_values)
    B_max_Q_values = np.array(B_max_Q_values)
    B_mean_Q_values = np.array(B_mean_Q_values)
    A_V_values = np.array(A_V_values)
    B_V_values = np.array(B_V_values)
    episode_rewards = np.array(episode_rewards)
    alpha_values = np.array(alpha_values)
    epsilon_values = np.array(epsilon_values)
    # print(bool_A_left.shape, Aleft_Q_values.shape, B_max_Q_values.shape, A_V_values.shape, B_V_values.shape)
    # return {"VA": A_V_values, "VB":B_V_values}, bool_A_left, Aleft_Q_values, B_max_Q_values
    return {"A_V": A_V_values, "B_V":B_V_values, "bool_A_left": bool_A_left, "Aleft_Q_values": Aleft_Q_values, "Aright_Q_values": Aright_Q_values, "B_max_Q_values": B_max_Q_values, "B_mean_Q_values": B_mean_Q_values, "episode_rewards": episode_rewards, "alpha_values": alpha_values, "epsilon_values": epsilon_values, "Q": Q, "V": V}





# entry of script
if __name__ == "__main__":
    VQ_alpha = 10
    init_factor = 1.0
    factor = init_factor
    num_of_episode = 300
    x_interval = int(num_of_episode/10)
    env_random_reward = True
    v_noise = 0.0
    # v_noise = 0.01

    # save_path = f"/storage/xue/repos/Reinforcement_Learning/images/ex6.7/Q_DQ_VQ/VQ_factor_{VQ_alpha}_E_{num_of_episode}"
    # os.makedirs(save_path, exist_ok=True)

    save_path = None

    note = f"factor={VQ_alpha}, n_episode={num_of_episode}"
    note = ""


    # method = ['Q-Learning', 'Expected_Sarsa', 'Action_Distribution', 'Double-Q']
    env = Env(0.1, 1, 10, env_random_reward)   

    # alpha_scope = [0.1, 0.001, 0.995]
    alpha_scope = [0.1, 0.001, 1.0]
    # start_epsilon = 0.1
    epsilon_scope = [0.1, 0.001, 1.0]
    # epsilon_scope = [0.1, 0.001, 0.995]
    gamma = 1.0

    output = TD_learning(env, method='Q-Learning', alpha_scope=alpha_scope, epsilon_scope=epsilon_scope, num_of_episode=num_of_episode, gamma=gamma, value_noise=v_noise)  

    # print(output['V'])
    # print("========")
    # print(output['Q'])

