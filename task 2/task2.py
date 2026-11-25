import pandas as pd
import numpy as np

# Read the original CSV
df = pd.read_csv("employees.csv")

# Record the original data quality counts
rows_before = len(df)
missing_before = int(df.isna().sum().sum())
duplicates_before = int(df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

# Remove commas from salary text, then convert it to a number
df["salary"] = (
    df["salary"]
    .astype("string")
    .str.replace(",", "", regex=False)
)
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

# Convert other numeric columns
for column in ["id", "age", "experience_years"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Trim extra spaces from text columns
for column in ["name", "department", "city", "attrition"]:
    df[column] = df[column].astype("string").str.strip()

# Fill missing salary with its median
salary_median = df["salary"].median()
df["salary"] = df["salary"].fillna(salary_median)

# Preserve missing experience as unknown and flag it for review
df["experience_years_missing"] = df["experience_years"].isna()

# Detect salary outliers using the IQR rule.
# Keep them in the data, but flag them for review.
q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df["salary_outlier_iqr"] = (
    (df["salary"] < lower_bound) |
    (df["salary"] > upper_bound)
)

# Make a before/after cleaning summary
cleaning_summary = pd.DataFrame({
    "measure": ["Rows", "Missing values", "Duplicate rows"],
    "before": [
        rows_before,
        missing_before,
        duplicates_before
    ],
    "after": [
        len(df),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum())
    ]
})

print("\n--- CLEANING SUMMARY ---")
print(cleaning_summary.to_string(index=False))

# Save the cleaned data and summary
df.to_csv("clean_data.csv", index=False)
cleaning_summary.to_csv("cleaning_summary.csv", index=False)

print("\nSaved clean_data.csv and cleaning_summary.csv")
print("Missing experience values are flagged in experience_years_missing.")
print("Salary outliers are flagged in salary_outlier_iqr and kept for review.")

# Calculate summary statistics for numeric columns
numeric_columns = ["salary", "age", "experience_years"]

statistics = df[numeric_columns].agg(
    ["mean", "median", "min", "max", "std"]
).T

print("\n--- NUMERIC SUMMARY STATISTICS ---")
print(statistics.to_string())

# Compare departments
department_summary = df.groupby("department").agg(
    employee_count=("id", "count"),
    average_salary=("salary", "mean"),
    median_salary=("salary", "median"),
    average_experience=("experience_years", "mean")
).sort_values("average_salary", ascending=False)

print("\n--- DEPARTMENT SUMMARY ---")
print(department_summary.to_string())

# Find the three strongest numeric correlations by absolute size
correlations = df[numeric_columns].corr()

upper_triangle = correlations.where(
    np.triu(np.ones(correlations.shape), k=1).astype(bool)
)

top_correlations = (
    upper_triangle.stack()
    .rename("correlation")
    .reset_index()
    .rename(columns={
        "level_0": "variable_1",
        "level_1": "variable_2"
    })
)

top_correlations["absolute_correlation"] = (
    top_correlations["correlation"].abs()
)

top_correlations = top_correlations.sort_values(
    "absolute_correlation",
    ascending=False
).head(3)

print("\n--- TOP NUMERIC CORRELATIONS ---")
if top_correlations.empty:
    print("There are not enough numeric columns to calculate correlations.")
else:
    print(top_correlations.to_string(index=False))

# Answer five business questions using values from the CSV
department_counts = df["department"].value_counts()
highest_count_department = department_counts.idxmax()

highest_salary_department = department_summary["average_salary"].idxmax()

attrition_rate = (
    df["attrition"].str.casefold().eq("yes")
    .groupby(df["department"])
    .mean()
    .mul(100)
)

city_median_salary = df.groupby("city")["salary"].median()
highest_attrition_department = attrition_rate.idxmax()

print("\n--- FIVE BUSINESS QUESTIONS AND ANSWERS ---")

print(
    f"1. Which department has the most employees? "
    f"{highest_count_department}: "
    f"{department_counts.max()} employees."
)

print(
    f"2. Which department has the highest average salary? "
    f"{highest_salary_department}: "
    f"{department_summary.loc[highest_salary_department, 'average_salary']:.2f}."
)

print("3. What is the attrition rate by department?")
print(attrition_rate.round(1).to_string() + "%")

print("4. What is the median salary by city?")
print(city_median_salary.round(2).to_string())

print(
    f"5. Which department has the highest attrition rate? "
    f"{highest_attrition_department}: "
    f"{attrition_rate.loc[highest_attrition_department]:.1f}%."
)

# Demonstrate merging a second small department-budget table.
# These are fictional example values; replace them with real data if provided.
budget_table = pd.DataFrame({
    "department": ["Sales", "IT", "HR", "Finance", "Marketing"],
    "monthly_budget_inr": [200000, 300000, 150000, 180000, 160000]
})

department_with_budget = pd.merge(
    department_summary.reset_index(),
    budget_table,
    on="department",
    how="left"
)

print("\n--- DEPARTMENT SUMMARY WITH EXAMPLE BUDGETS ---")
print(department_with_budget.to_string(index=False))

# Bonus: pivot table showing department and attrition counts
attrition_pivot = pd.pivot_table(
    df,
    index="department",
    columns="attrition",
    values="id",
    aggfunc="count",
    fill_value=0
)

print("\n--- BONUS: DEPARTMENT BY ATTRITION PIVOT TABLE ---")
print(attrition_pivot.to_string())