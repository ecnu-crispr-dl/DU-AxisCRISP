import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from common_def import *
NUM_SAMPLE = None
FEATURES = "v2"
FREQ_THRESHOLD = 0.3

def analyze_longtail_distribution(y_arrays, samples, threshold=0.2):
    """分析长尾分布特征"""
    num_samples = len(samples)
    freq_counts = []
    tail_ratios = []
    
    for sample in samples[:num_samples]:
        y = y_arrays.loc[sample].values
        sorted_freqs = np.sort(y)[::-1]
        freq_counts.append(sorted_freqs)
        
        top_10_percent = int(len(sorted_freqs) * 0.1)
        if top_10_percent > 0:
            top_sum = np.sum(sorted_freqs[:top_10_percent])
            total_sum = np.sum(sorted_freqs)
            tail_ratio = top_sum / total_sum if total_sum > 0 else 0
            tail_ratios.append(tail_ratio)
    
    print(f"   总Indel数: 705")
    print(f"   平均长尾比例（前10%占比）: {np.mean(tail_ratios):.3f} ± {np.std(tail_ratios):.3f}")
    print(f"   长尾程度: {'极强' if np.mean(tail_ratios) > 0.8 else '强' if np.mean(tail_ratios) > 0.6 else '中等'}")
    
    all_freqs = np.concatenate(freq_counts)
    print(f"   频率分布特征:")
    print(f"     - 最大值: {np.max(all_freqs):.4f}")
    print(f"     - 中位数: {np.median(all_freqs):.4f}")
    print(f"     - 99%分位数: {np.percentile(all_freqs, 99):.4f}")
    print(f"     - 95%分位数: {np.percentile(all_freqs, 95):.4f}")



with open(indels_sorted_path, "rb") as f:
    ALL_INDELS = pkl.load(f)
X, y, samples, sequences = load_delete_data(
    filepath=train_file_path, num_samples=NUM_SAMPLE, fractions=True, indel_list=ALL_INDELS
)
X = X.loc[:, FEATURE_SETS[FEATURES]]
prior = compute_prior_stable(y)
A = make_logit_adjustment(prior, tau=1, device=DEVICE)

print(f"   样本数: {len(samples)}")
print(f"   特征数: {X.shape[1]}")
print(f"   Indel数: {len(ALL_INDELS)}")
print(f"   序列特征维度: {sequences.shape[1]}")

analyze_longtail_distribution(y, samples, FREQ_THRESHOLD)


