import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

RESULTS_FOLDER = './src/test_results_fcc_1/'
NUM_BINS = 1000
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
REBUF_P = 4.3
SMOOTH_P = 1

SCHEME = 'sim_oboempc'
LABEL = 'Oboe+MPC'
LW = 3.1

def main():
    time_all = {}
    raw_reward_all = {}

    time_all[SCHEME] = {}
    raw_reward_all[SCHEME] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:
        if SCHEME not in log_file:
            continue

        time_ms = []
        reward = []

        with open(os.path.join(RESULTS_FOLDER, log_file), 'r') as f:
            for line in f:
                parse = line.split()
                if len(parse) <= 1:
                    break
                time_ms.append(float(parse[0]))
                reward.append(float(parse[-1]))

        key = log_file[len('log_' + SCHEME + '_'):]
        time_all[SCHEME][key] = time_ms
        raw_reward_all[SCHEME][key] = reward

    reward_all = []
    for l in time_all[SCHEME]:
        if len(time_all[SCHEME][l]) >= VIDEO_LEN:
            avg_reward = np.mean(raw_reward_all[SCHEME][l][1:VIDEO_LEN])
            reward_all.append(avg_reward)

    mean_reward = np.mean(reward_all)
    print(f"{SCHEME}: Average QoE = {mean_reward:.3f}")

    # Plot CDF
    plt.rcParams['axes.labelsize'] = 16
    font = {'size': 16}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    plt.subplots_adjust(left=0.17, bottom=0.19, right=0.97, top=0.96)

    values, base = np.histogram(reward_all, bins=NUM_BINS)
    cumulative = np.cumsum(values) / np.sum(values)
    ax.plot(base[:-1], cumulative, '-', color='#6631A0', lw=LW, label=LABEL)

    ax.legend(framealpha=1, frameon=False, fontsize=14)
    plt.ylim(0., 1.)
    plt.xlim(-0.5, 4.3)
    plt.ylabel('CDF')
    plt.xlabel('Average QoE')
    os.makedirs('details', exist_ok=True)
    plt.savefig('details/cdf_oboempc_only.png')
    plt.show()

if __name__ == '__main__':
    main()
