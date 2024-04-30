import matplotlib.pyplot as plt

def plot_prob_A_left(prob_Q_A_left, prob_Q2_A_left, prob_E_A_left, prob_AD_A_left, prob_VQ_A_left, num_of_episode, std=False, note='', save_path=None):
    plt.ylabel('Probability of taking action left from A')
    plt.xlabel('Episodes')
    x_ticks = np.arange(0,num_of_episode +1, x_interval)
    y_ticks = np.arange(0,1.1,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks,['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
    if prob_Q_A_left is not None:
        plt.plot(range(num_of_episode), prob_Q_A_left.mean(axis=0), '-',label='Q Learning')
        if std: plt.fill_between(range(num_of_episode), prob_Q_A_left.mean(axis=0) - prob_Q_A_left.std(axis=0)/2, prob_Q_A_left.mean(axis=0) + prob_Q_A_left.std(axis=0)/2, alpha=0.2)
    if prob_Q2_A_left is not None:
        plt.plot(range(num_of_episode), prob_Q2_A_left.mean(axis=0), '-',label='Double Q-Learning')
        if std: plt.fill_between(range(num_of_episode), prob_Q2_A_left.mean(axis=0) - prob_Q2_A_left.std(axis=0)/2, prob_Q2_A_left.mean(axis=0) + prob_Q2_A_left.std(axis=0)/2, alpha=0.2)
    if prob_E_A_left is not None:
        plt.plot(range(num_of_episode), prob_E_A_left.mean(axis=0), '-',label='Sarsa')
        if std: plt.fill_between(range(num_of_episode), prob_E_A_left.mean(axis=0) - prob_E_A_left.std(axis=0)/2, prob_E_A_left.mean(axis=0) + prob_E_A_left.std(axis=0)/2, alpha=0.2)
    if prob_AD_A_left is not None:
        plt.plot(range(num_of_episode), prob_AD_A_left.mean(axis=0), '-',label='Action Distribution')
        if std: plt.fill_between(range(num_of_episode), prob_AD_A_left.mean(axis=0) - prob_AD_A_left.std(axis=0)/2, prob_AD_A_left.mean(axis=0) + prob_AD_A_left.std(axis=0)/2, alpha=0.2)
    if prob_VQ_A_left is not None:
        plt.plot(range(num_of_episode), prob_VQ_A_left.mean(axis=0), '-',label='VQ Learning')
        if std: plt.fill_between(range(num_of_episode), prob_VQ_A_left.mean(axis=0) - prob_VQ_A_left.std(axis=0)/2, prob_VQ_A_left.mean(axis=0) + prob_VQ_A_left.std(axis=0)/2, alpha=0.2)
    plt.plot(np.ones(num_of_episode) * 0.05, label='Optimal')
    plt.title('Comparison of probability of left actions from A' + note)
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(f'{save_path}/prob_A_left.png', dpi=200)
    plt.show()
    plt.close()


def plot_Q_A(A_Q_lst, A_Q2_lst, A_AD_lst, A_E_lst, A_VQ_lst, y_label="Q(A, left)", std=False, note='', save_path=None):
    import matplotlib.pyplot as plt
    plt.ylabel(y_label)
    plt.xlabel('Episodes')
    x_ticks = np.arange(0,num_of_episode +1, x_interval)
    # y_ticks = np.arange(0,1.1,0.1)
    plt.xticks(x_ticks)
    # plt.yticks(y_ticks,['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
    if A_Q_lst is not None:
        plt.plot(range(num_of_episode), A_Q_lst.mean(axis=0), '-',label='Q Learning')
        if std: plt.fill_between(range(num_of_episode), A_Q_lst.mean(axis=0) - A_Q_lst.std(axis=0)/2, A_Q_lst.mean(axis=0) + A_Q_lst.std(axis=0)/2, alpha=0.2)
    if A_Q2_lst is not None:
        plt.plot(range(num_of_episode), A_Q2_lst.mean(axis=0), '-',label='Double Q-Learning')
        if std: plt.fill_between(range(num_of_episode), A_Q2_lst.mean(axis=0) - A_Q2_lst.std(axis=0)/2, A_Q2_lst.mean(axis=0) + A_Q2_lst.std(axis=0)/2, alpha=0.2)
    if A_AD_lst is not None:
        plt.plot(range(num_of_episode), A_AD_lst.mean(axis=0), '-',label='Action Distribution')
        if std: plt.fill_between(range(num_of_episode), A_AD_lst.mean(axis=0) - A_AD_lst.std(axis=0)/2, A_AD_lst.mean(axis=0) + A_AD_lst.std(axis=0)/2, alpha=0.2)
    if A_E_lst is not None:
        plt.plot(range(num_of_episode), A_E_lst.mean(axis=0), '-',label='Expected Sarsa')
        if std: plt.fill_between(range(num_of_episode), A_E_lst.mean(axis=0) - A_E_lst.std(axis=0)/2, A_E_lst.mean(axis=0) + A_E_lst.std(axis=0)/2, alpha=0.2)
    if A_VQ_lst is not None:
        plt.plot(range(num_of_episode), A_VQ_lst.mean(axis=0), '-',label='VQ Learning')
        if std: plt.fill_between(range(num_of_episode), A_VQ_lst.mean(axis=0) - A_VQ_lst.std(axis=0)/2, A_VQ_lst.mean(axis=0) + A_VQ_lst.std(axis=0)/2, alpha=0.2)
    # plt.plot(np.ones(num_of_episode) * 0.05, label='Optimal')
    plt.title(f'Comparison of {y_label}, {note}')
    plt.legend()
    plt.grid()
    if save_path is not None:
        # y_label = y_label.replace(',', '_')
        plt.savefig(f'{save_path}/{y_label}.png', dpi=200)
    plt.show()
    plt.close()

def plot_Q_B(B_Q_lst, B_Q2_lst, B_AD_lst, B_E_lst, B_VQ_lst, y_label='', std=False, note='', save_path=None):
    import matplotlib.pyplot as plt
    plt.ylabel(y_label)
    plt.xlabel('Episodes')
    x_ticks = np.arange(0,num_of_episode +1, x_interval)
    # y_ticks = np.arange(0,1.1,0.1)
    plt.xticks(x_ticks)
    # plt.yticks(y_ticks,['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
    if B_Q_lst is not None:
        plt.plot(range(num_of_episode), B_Q_lst.mean(axis=0), '-',label='Q Learning')
        if std: plt.fill_between(range(num_of_episode), B_Q_lst.mean(axis=0) - B_Q_lst.std(axis=0)/2, B_Q_lst.mean(axis=0) + B_Q_lst.std(axis=0)/2, alpha=0.2)
    if B_Q2_lst is not None:
        plt.plot(range(num_of_episode), B_Q2_lst.mean(axis=0), '-',label='Double Q-Learning')
        if std: plt.fill_between(range(num_of_episode), B_Q2_lst.mean(axis=0) - B_Q2_lst.std(axis=0)/2, B_Q2_lst.mean(axis=0) + B_Q2_lst.std(axis=0)/2, alpha=0.2)
    if B_AD_lst is not None:
        plt.plot(range(num_of_episode), B_AD_lst.mean(axis=0), '-',label='Action Distribution')
        if std: plt.fill_between(range(num_of_episode), B_AD_lst.mean(axis=0) - B_AD_lst.std(axis=0)/2, B_AD_lst.mean(axis=0) + B_AD_lst.std(axis=0)/2, alpha=0.2)
    if B_E_lst is not None:
        plt.plot(range(num_of_episode), B_E_lst.mean(axis=0), '-',label='Expected Sarsa')
        if std: plt.fill_between(range(num_of_episode), B_E_lst.mean(axis=0) - B_E_lst.std(axis=0)/2, B_E_lst.mean(axis=0) + B_E_lst.std(axis=0)/2, alpha=0.2)
    if B_VQ_lst is not None:
        plt.plot(range(num_of_episode), B_VQ_lst.mean(axis=0), '-',label='VQ Learning')
        if std: plt.fill_between(range(num_of_episode), B_VQ_lst.mean(axis=0) - B_VQ_lst.std(axis=0)/2, B_VQ_lst.mean(axis=0) + B_VQ_lst.std(axis=0)/2, alpha=0.2)
    # plt.plot(np.ones(num_of_episode) * 0.05, label='Optimal')
    plt.title(f'Comparison of {y_label}, {note}')
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(f'{save_path}/{y_label}.png', dpi=200)
    plt.show()
    plt.close()

def plot_V(V_Q_lst, V_Q2_lst, V_AD_lst, V_E_lst, V_VQ_lst, y_label, std=False, note='', save_path=None):
    import matplotlib.pyplot as plt
    plt.ylabel(f'{y_label}')
    plt.xlabel('Episodes')
    x_ticks = np.arange(0,num_of_episode +1, x_interval)
    # y_ticks = np.arange(0,1.1,0.1)
    plt.xticks(x_ticks)
    # plt.yticks(y_ticks,['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
    if V_Q_lst is not None:
        plt.plot(range(num_of_episode), V_Q_lst.mean(axis=0), '-',label='Q Learning')
        if std: plt.fill_between(range(num_of_episode), V_Q_lst.mean(axis=0) - V_Q_lst.std(axis=0)/2, V_Q_lst.mean(axis=0) + V_Q_lst.std(axis=0)/2, alpha=0.2)
    if V_Q2_lst is not None:
        plt.plot(range(num_of_episode), V_Q2_lst.mean(axis=0), '-',label='Double Q-Learning')
        if std: plt.fill_between(range(num_of_episode), V_Q2_lst.mean(axis=0) - V_Q2_lst.std(axis=0)/2, V_Q2_lst.mean(axis=0) + V_Q2_lst.std(axis=0)/2, alpha=0.2)
    if V_AD_lst is not None:
        plt.plot(range(num_of_episode), V_AD_lst.mean(axis=0), '-',label='Action Distribution')
        if std: plt.fill_between(range(num_of_episode), V_AD_lst.mean(axis=0) - V_AD_lst.std(axis=0)/2, V_AD_lst.mean(axis=0) + V_AD_lst.std(axis=0)/2, alpha=0.2)
    if V_E_lst is not None:
        plt.plot(range(num_of_episode), V_E_lst.mean(axis=0), '-',label='Expected Sarsa')
        if std: plt.fill_between(range(num_of_episode), V_E_lst.mean(axis=0) - V_E_lst.std(axis=0)/2, V_E_lst.mean(axis=0) + V_E_lst.std(axis=0)/2, alpha=0.2)
    if V_VQ_lst is not None:
        plt.plot(range(num_of_episode), V_VQ_lst.mean(axis=0), '-',label='VQ Learning')
        if std: plt.fill_between(range(num_of_episode), V_VQ_lst.mean(axis=0) - V_VQ_lst.std(axis=0)/2, V_VQ_lst.mean(axis=0) + V_VQ_lst.std(axis=0)/2, alpha=0.2)
    # plt.plot(np.ones(num_of_episode) * 0.05, label='Optimal')
    plt.title(f'Comparison of {y_label}, {note}')
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(f'{save_path}/{y_label}.png', dpi=200)
    plt.show()    
    plt.close()

def plot_V_Q(V_Q_lst=None, Q_Q_lst=None, V_Q2_lst=None, Q_Q2_lst=None, V_AD_lst=None, Q_AD_lst=None, V_E_lst=None, Q_E_lst=None, V_VQ_lst=None, Q_VQ_lst=None, y_label='', note='', save_path=None):
    import matplotlib.pyplot as plt
    plt.ylabel(y_label)
    plt.xlabel('Episodes')
    x_ticks = np.arange(0,num_of_episode +1, x_interval)
    # y_ticks = np.arange(0,1.1,0.1)
    plt.xticks(x_ticks)
    # plt.yticks(y_ticks,['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
    if V_Q_lst is not None:
        plt.plot(range(num_of_episode), V_Q_lst, '-',label='V - Q Learning')
        plt.plot(range(num_of_episode), Q_Q_lst, '-',label='Q - Q Learning')
    if V_Q2_lst is not None:
        plt.plot(range(num_of_episode), V_Q2_lst, '-',label='V - Double Q-Learning')
        plt.plot(range(num_of_episode), Q_Q2_lst, '-',label='Q - Double Q-Learning')
    if V_AD_lst is not None:
        plt.plot(range(num_of_episode), V_AD_lst, '-',label='V - Action Distribution')
        plt.plot(range(num_of_episode), Q_AD_lst, '-',label='Q - Action Distribution')
    if V_E_lst is not None:
        plt.plot(range(num_of_episode), V_E_lst, '-',label='V - Expected Sarsa')
        plt.plot(range(num_of_episode), Q_E_lst, '-',label='Q - Expected Sarsa')
    if V_VQ_lst is not None:
        plt.plot(range(num_of_episode), V_VQ_lst, '-',label='V - VQ Learning')
        plt.plot(range(num_of_episode), Q_VQ_lst, '-',label='Q - VQ Learning')
    # plt.plot(np.ones(num_of_episode) * 0.05, label='Optimal')
    plt.title(f'Comparison of {y_label}, {note}')
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(f'{save_path}/{y_label}.png', dpi=200)
    plt.show()
    plt.close()