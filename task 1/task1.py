import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from pathlib import Path

df = pd.read_csv("employees.csv")

#
print ("\n--- Before Removing Duplicates ------\n")
duplicates_count = df.duplicated().sum()

print(f"counts of duplicates in the dataset:{duplicates_count}")

df = df.drop_duplicates()

duplicates_count1 = df.duplicated().sum()

print("\n----- After Removing Duplicates -----\n")

print(f"counts of duplicates in the dataset:{duplicates_count1}")

null_values_count = df.isna().sum()

print("\n------ Null Values in the Dataset -------\n")

print(f"counts of null valuesin the dataset:{null_values_count}")

df["salary"] = (df["salary"] .astype(str).str.replace(",","",regex=False))

df["salary"] = pd.to_numeric(df["salary"],errors="coerce")

df["salary"] = df["salary"].fillna(df["salary"].median())



mask = (df["id"] == 1019) & (df["experience_years"].isna())
df.loc[mask, "experience_years"] = 7


null_values_count1 = df.isna().sum()

print("\n------ After HandlingNull Values in the Dataset -------\n")

print(f"counts of null valuesin the dataset:{null_values_count1}    ")




FIGURE_DIR = Path("reports") / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SALARY_UNIT = "INR"
MAIN_COLOR = "#2F6690"

# Reuse the same color for each department across charts
departments = df["department"].dropna().unique()
cmap = plt.get_cmap("tab10")
department_colors = {
    department: cmap(i % 10)
    for i, department in enumerate(departments)
}

def save_chart(fig, filename, insight):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Insight: {insight}")
    plt.close(fig)


# ------- BAR CHART --------------
counts = df["department"].value_counts().sort_values(ascending=False)




fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [department_colors[dept] for dept in counts.index]
ax.bar(counts.index, counts.values, color=bar_colors)

ax.set_title("Employee Count by Department")
ax.set_xlabel("Department")
ax.set_ylabel("Number of Employees")

top_department = counts.idxmax()
insight = (
    f"{top_department} has the most employees "
    f"({counts.max()}) in this dataset."
)

save_chart(fig, "employee_count_by_department.png", insight)


# histogram of salaries
fig,ax = plt.subplots(figsize=(8,5))
counts,bins,patches = ax.hist(df["salary"],bins =5, edgecolor = "black" )
cmap = plt.cm.viridis
norm = plt.Normalize(bins[0],bins[-1])
for patch , left,right in zip(patches,bins[:-1],bins[1:]):
    middle_salary = (left+right)/2
    patch.set_facecolor(cmap(norm(middle_salary)))



ax.set_title("Salary Distribution")
ax.set_xlabel(f"salary({SALARY_UNIT})")
ax.set_ylabel(f"Number of Employees")

most_common_bin = bins[np.argmax(counts)]

insight = (f"the most common salary distribution is between {most_common_bin} to {bins[np.argmax(counts) + 1]} ")

save_chart(fig,"salary_distribution.png",insight)


# SCATTER PLOT

plot_data = df.dropna(
    subset=["experience_years", "salary", "department"]
)

fig, ax = plt.subplots(figsize=(8, 5))

for department in departments:
    rows = plot_data[plot_data["department"] == department]
    ax.scatter(
        rows["experience_years"],
        rows["salary"],
        color=department_colors[department],
        label=department,
        alpha=0.8
    )

ax.set_title("Experience vs Salary by Department")
ax.set_xlabel("Experience (years)")
ax.set_ylabel(f"Salary ({SALARY_UNIT})")
ax.legend(title="Department")

correlation = plot_data["experience_years"].corr(plot_data["salary"])
direction = "positive" if correlation >= 0 else "negative"
insight = (
    f"Experience and salary have a {direction} association "
    f"in this sample (correlation: {correlation:.2f}). "
    "This does not prove that one causes the other."
)

save_chart(fig, "experience_vs_salary.png", insight)


# LINE CHART 
average_salary = (
    df.groupby("department")["salary"]
    .mean()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    average_salary.index,
    average_salary.values,
    color=MAIN_COLOR,
    marker="o"
)

ax.set_title("Average Salary by Department, Sorted")
ax.set_xlabel("Department (sorted by average salary)")
ax.set_ylabel(f"Average Salary ({SALARY_UNIT})")
ax.tick_params(axis="x", rotation=25)

lowest_department = average_salary.idxmin()
highest_department = average_salary.idxmax()

insight = (
    f"{lowest_department} has the lowest average salary, while "
    f"{highest_department} has the highest in this dataset."
)

save_chart(fig, "average_salary_sorted.png", insight)


# 5. BOX PLOT
box_departments = sorted(df["department"].dropna().unique())
box_data = [
    df.loc[df["department"] == department, "salary"].dropna()
    for department in box_departments
]

fig, ax = plt.subplots(figsize=(8, 5))
boxes = ax.boxplot(
    box_data,
    tick_labels=box_departments,
    patch_artist=True
)

for box, department in zip(boxes["boxes"], box_departments):
    box.set_facecolor(department_colors[department])

ax.set_title("Salary Spread by Department")
ax.set_xlabel("Department")
ax.set_ylabel(f"Salary ({SALARY_UNIT})")
ax.tick_params(axis="x", rotation=25)

department_medians = df.groupby("department")["salary"].median()
highest_median_dept = department_medians.idxmax()
lowest_median_dept = department_medians.idxmin()

insight = (
    f"{highest_median_dept} has the highest median salary; "
    f"{lowest_median_dept} has the lowest."
)

save_chart(fig, "salary_boxplot_by_department.png", insight)
