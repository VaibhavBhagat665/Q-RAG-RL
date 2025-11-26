"""
Compare model performance against baseline methods.
Shows cost and safety metrics comparison.
"""
import pandas as pd

print("=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)

method_comparison = pd.read_csv('results/method_comparison_results.csv')
industry_test = pd.read_csv('results/industry_test_results.csv')

print("\nMethod Comparison (24-hour simulation):")
print("-" * 80)
for _, row in method_comparison.iterrows():
    print(f"\n{row['Method']}:")
    print(f"  Cost: ${row['Cost']/24:.4f}/hour")
    print(f"  Violations: {row['Violations']}/24")
    print(f"  Safety: {(24-row['Violations'])/24*100:.1f}%")

qragrl_cost = method_comparison[method_comparison['Method'] == 'Q-RAG-RL (Yours)']['Cost'].values[0]
baseline_cost = method_comparison[method_comparison['Method'] == 'No Control']['Cost'].values[0]

savings = (baseline_cost - qragrl_cost) / baseline_cost * 100

print(f"\nCost Savings:")
print(f"  Q-RAG-RL: ${qragrl_cost/24:.4f}/hour")
print(f"  Baseline: ${baseline_cost/24:.4f}/hour")
print(f"  Savings: {savings:.2f}%")

industry_good = industry_test[industry_test['avg_cost'] < 1.0]
print(f"\nIndustry Performance:")
print(f"  Average cost: ${industry_good['avg_cost'].mean():.4f}/hour")
print(f"  Safety rate: {industry_good['safety_rate'].mean():.1f}%")

print("\n" + "=" * 80)
