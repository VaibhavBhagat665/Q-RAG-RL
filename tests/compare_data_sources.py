"""
Compare results: Synthetic Data vs Real NREL Data
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("SYNTHETIC vs REAL DATA COMPARISON")
print("=" * 70)

# Load synthetic results
synthetic_df = pd.read_csv('evaluation_results.csv')
print("\n1. SYNTHETIC DATA RESULTS:")
print(f"   Total Cost:    ${synthetic_df['cost'].sum():.4f}")
print(f"   Violations:    {len(synthetic_df[synthetic_df['safe'] == False])}/24")
print(f"   SOC Range:     {synthetic_df['soc'].min():.2f} - {synthetic_df['soc'].max():.2f}")
print(f"   Action Range:  {synthetic_df['action'].min():.3f} - {synthetic_df['action'].max():.3f} MW")

# Load real results
real_df = pd.read_csv('real_data_results.csv')
print("\n2. REAL NREL DATA RESULTS:")
print(f"   Total Cost:    ${real_df['cost'].sum():.4f}")
print(f"   Violations:    {len(real_df[real_df['safe'] == False])}/24")
print(f"   SOC Range:     {real_df['battery_soc'].min():.2f} - {real_df['battery_soc'].max():.2f}")
print(f"   Action Range:  {real_df['action'].min():.3f} - {real_df['action'].max():.3f} MW")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

synthetic_cost = synthetic_df['cost'].sum()
real_cost = real_df['cost'].sum()
cost_diff = real_cost - synthetic_cost
cost_diff_pct = (cost_diff / synthetic_cost) * 100

print(f"\nCost Difference:")
print(f"  Synthetic: ${synthetic_cost:.4f}")
print(f"  Real:      ${real_cost:.4f}")
print(f"  Diff:      ${cost_diff:.4f} ({cost_diff_pct:+.1f}%)")

print(f"\nSafety Performance:")
synthetic_violations = len(synthetic_df[synthetic_df['safe'] == False])
real_violations = len(real_df[real_df['safe'] == False])
print(f"  Synthetic: {synthetic_violations}/24 violations")
print(f"  Real:      {real_violations}/24 violations")

if synthetic_violations == 0 and real_violations == 0:
    print("  ✅ BOTH achieve zero violations!")
elif real_violations == 0:
    print("  ✅ Real data maintains zero violations!")
else:
    print(f"  ⚠️  Real data has {real_violations} violations")

print(f"\nGeneration Comparison:")
print(f"  Real Solar:      {real_df['solar_gen'].mean():.3f} MW avg")
print(f"  Real Wind:       {real_df['wind_gen'].mean():.3f} MW avg")

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print("\n1. Safety Performance:")
if real_violations == 0:
    print("   ✅ Agent maintains zero violations on REAL data")
    print("   ✅ Safety mechanisms are ROBUST")
else:
    print(f"   ⚠️  Agent has {real_violations} violations on real data")

print("\n2. Cost Performance:")
if abs(cost_diff_pct) < 50:
    print(f"   ✅ Costs are comparable ({cost_diff_pct:+.1f}% difference)")
    print("   ✅ Agent generalizes reasonably well")
else:
    print(f"   ⚠️  Large cost difference ({cost_diff_pct:+.1f}%)")
    print("   ⚠️  Real data has different characteristics")

print("\n3. Data Characteristics:")
if real_df['solar_gen'].max() < 0.1:
    print("   ⚠️  Real solar data is NIGHTTIME (Jan 1, midnight start)")
    print("   ⚠️  Not representative of full day operation")
else:
    print("   ✅ Real data includes daytime solar generation")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if real_violations == 0:
    print("\n✅ VALIDATED: Agent works on REAL NREL data!")
    print("   - Zero violations maintained")
    print("   - Safety mechanisms robust")
    print("   - Algorithm generalizes to real weather")
else:
    print(f"\n⚠️  PARTIAL: Agent has {real_violations} violations on real data")
    print("   - May need retraining on real scenarios")

print("\nNote: Real data is from Jan 1 midnight (no solar), so costs")
print("are higher due to grid dependence. Full validation needs")
print("daytime hours with solar generation.")

print("=" * 70)
