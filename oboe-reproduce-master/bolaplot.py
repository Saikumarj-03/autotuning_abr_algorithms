import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_FOLDER = './src/test_results/'
VIDEO_LEN = 48
NUM_BINS = 1000

# Collect average bitrate per trace
bitrate_oboebola = {}
bitrate_bolabase = {}

for log_file in os.listdir(RESULTS_FOLDER):
    full_path = os.path.join(RESULTS_FOLDER, log_file)
    with open(full_path, 'r') as f:
        chunk_bitrates = []
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                break
            chunk_bitrates.append(float(parts[1]))
        if len(chunk_bitrates) >= VIDEO_LEN:
            avg_bitrate = np.mean(chunk_bitrates[:VIDEO_LEN])
            trace_id = log_file.split('_')[-1]
            if 'sim_oboebola' in log_file:
                bitrate_oboebola[trace_id] = avg_bitrate
            elif 'sim_bolabase' in log_file:
                bitrate_bolabase[trace_id] = avg_bitrate

# Calculate % improvement for traces with both results
improvements = []
for trace_id in bitrate_oboebola:
    if trace_id in bitrate_bolabase:
        base = bitrate_bolabase[trace_id]
        oboe = bitrate_oboebola[trace_id]
        improvement = (oboe - base) / base * 1000000000
        improvements.append(improvement)

# Plotting the CDF
plt.figure(figsize=(4.5, 3))
values, base = np.histogram(improvements, bins=NUM_BINS)
cdf = np.cumsum(values) / np.sum(values) * 100  # convert to percentage

plt.plot(base[:-1], cdf, 'r-.', linewidth=2.5, label='Avg. Bitrate')

# Formatting to match the provided style
plt.xlim(-5, 5)
plt.ylim(0, 100)
plt.xlabel('Perc. Imp. in Avg. Bitrate over BOLA')
plt.ylabel('CDF (Perc. of sessions)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=10)

# Add arrow annotation
plt.annotate('Better',
             xy=(12, 60),
             xytext=(18, 60),
             arrowprops=dict(arrowstyle='->', lw=1.5),
             fontsize=10,
             ha='left')

# Add subplot label
plt.text(-5, -12, '(a)', fontsize=11)

plt.tight_layout()
plt.savefig('percent_improvement_bitrate_cdf.png', dpi=300)
plt.show()
