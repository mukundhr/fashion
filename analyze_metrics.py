import pandas as pd
import re

# Original version metrics
print('=== ORIGINAL VERSION (with data leakage) ===')
orig = pd.read_csv('reports/model_comparison.csv')
print(orig.to_string(index=False))
print()

# Fixed version metrics  
print('=== FIXED VERSION (no data leakage) ===')
fixed = pd.read_csv('reports_fixed/model_comparison.csv')
print(fixed.to_string(index=False))
print()

# Best model details from fixed version report
with open('reports_fixed/fashion_classification_report.html', 'r') as f:
    content = f.read()
# Find the best model line
m = re.search(r'\[BEST\] Best Model: (.*?) with Test Accuracy: ([\d.]+)', content)
if m:
    print(f'Best model: {m.group(1)}, Test Accuracy: {m.group(2)}')

# Also print the full model comparison from report
comp_match = re.search(r'<h2>Model Comparison</h2>(.*?)</div>', content, re.DOTALL)
if comp_match:
    print("\nModel comparison from HTML report:")
    print(comp_match.group(1)[:500])