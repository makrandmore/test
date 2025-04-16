import numpy as np
from sklearn.datasets import load_diabetes
from scipy.stats import ttest_ind

X, y = load_diabetes(return_X_y=True)
bmi = X[:, 2]

median_bmi = np.median(bmi)
low_bmi_group = y[bmi < median_bmi]
high_bmi_group = y[bmi >= median_bmi]

t_stat, p_val = ttest_ind(low_bmi_group, high_bmi_group)

print("T-Statistic:", t_stat)
print("P-Value:", p_val)

if p_val < 0.05:
    print("Reject null hypothesis: Significant difference in means")
else:
    print("Fail to reject null hypothesis: No significant difference")
