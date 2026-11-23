import math
import statistics


def get_arithmetic_metrics(nums):
    count = len(nums)
    total = sum(nums)
    return {
        "count": count,
        "sum": total,
        "mean_manual": total / count,
        "min": min(nums),
        "max": max(nums),
    }


def get_statistical_metrics(nums):
    mean_val = statistics.mean(nums)
    median_val = statistics.median(nums)
    sample_std = statistics.stdev(nums)
    pop_std = math.sqrt(sum((x - mean_val) ** 2 for x in nums) / len(nums))
    return {
        "mean": mean_val,
        "median": median_val,
        "sample_std": round(sample_std, 2),
        "pop_std": round(pop_std, 2),
    }


scores = [10, 20, 30, 40, 50]
print("Input Data:", scores)

arith_res = get_arithmetic_metrics(scores)
print("\n--- Arithmetic Metrics ---")
for label, val in arith_res.items():
    print(f"{label}: {val}")

stat_res = get_statistical_metrics(scores)
print("\n--- Statistical Metrics ---")
for label, val in stat_res.items():
    print(f"{label}: {val}")