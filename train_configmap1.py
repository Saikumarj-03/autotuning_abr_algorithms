import numpy as np
import os
import ast
import fixed_env as env
import load_trace
import mpc

S_INFO = 5
S_LEN = 8
A_DIM = 6

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0

REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
CONFIGMAP_PATH = 'configmap_mpc.py'

def single_step(mu, sigma):
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(mu, sigma)
    reward_all = []
    for mpc_discount in range(10):
        reward_arr = []
        discount = mpc_discount / 10. + 0.1
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw)
        ccmpc = mpc.mpc()
        time_stamp = 0
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        r_batch = []
        state = np.zeros((S_INFO, S_LEN))
        video_count = 0
        while True:
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)
            time_stamp += delay + sleep_time
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            r_batch.append(reward)
            last_bit_rate = bit_rate
            state = np.roll(state, -1, axis=1)
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state[4, -1] = float(delay) / M_IN_K
            harmonic_bandwidth = np.sum(state[3] * state[4]) / np.sum(state[4]) * 1000.
            future_bandwidth = harmonic_bandwidth * discount
            bit_rate = ccmpc.run(future_bandwidth, buffer_size, last_bit_rate)
            if end_of_video:
                reward_arr.append(np.sum(r_batch))
                r_batch.clear()
                state = np.zeros((S_INFO, S_LEN))
                video_count += 1
                if video_count >= len(all_file_names):
                    break
        reward_all.append(np.array(reward_arr))
    reward_all = np.array(reward_all)
    return np.round(np.argmax(reward_all, axis=0) / 10. + 0.1, 1)

# Load existing configmap
with open(CONFIGMAP_PATH, 'r') as f:
    content = f.read()
    start = content.find('{')
    end = content.rfind('}') + 1
    existing_configmap = ast.literal_eval(content[start:end])

# Append new entries (mean 6.0 to 10.0 Mbps)
for mu in range(60, 101):  # mean from 6.0 to 10.0
    for sigma in range(0, 40):  # std from 0.0 to 4.0
        mu_ = mu / 10.
        sigma_ = sigma / 10.
        print(f"Processing mu={mu_}, sigma={sigma_}")
        best_params = single_step(mu_, sigma_)
        existing_configmap[(int(mu_ * 1000), int(sigma_ * 1000))] = list(best_params)

# Save the updated configmap
with open(CONFIGMAP_PATH, 'w') as f:
    f.write('configmap_mpc_oboe_900 = ')
    print(existing_configmap, file=f)

print("✅ Configmap updated and saved.")
