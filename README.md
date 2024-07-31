# STATS FOR DATA SCIENCE

**A Package for Data Science and Statistics from an OOP Perspective.**
This package is designed to adapt your data and metadata to an AnnData structure, leveraging its features for comprehensive data analysis. It will provide descriptive statistics, allow for correlation analysis, conduct t-tests between variables, perform ANOVAs, and create various types of plots.

## Overview of `Stastics` Class

### Constructor
- **`__init__(self, data, metadata)`**
  - Initializes the `Stastics` object with `data` and `metadata`.
  - Removes duplicated columns from metadata.
  - Identifies the type of each column in metadata (categorical or continuous).
  - Creates an initial description of the data.

### Methods

- **`__identify_column_type(self, series)`**
  - Identifies whether a column in the metadata is categorical or continuous based on the ratio of unique values.

- **`do_description(self, name="All", subset="All")`**
  - Generates and stores descriptive statistics for observations in the `AnnData` object.
  - Allows for optional subsetting of data.

- **`filter(self, var_name, subgroup="DROPNAS")`**
  - Filters observations based on a metadata or data column.
  - Can remove missing values or keep only specified values.

- **`find_var(self, var, adata_df=None)`**
  - Finds and returns specified variables from the `AnnData` object or its DataFrame representation.

- **`__check_normality(self, values, condition=None)`**
  - Checks if data follows a normal distribution, either for all data or within specific conditions.

- **`__hodges_lehmann_estimator(self, group1, group2)`**
  - Calculates the Hodges-Lehmann estimator for two independent samples.

- **`comparisons_1_1(self, target, condition_name)`**
  - Performs statistical comparisons (T-test or Mann-Whitney U test) between two groups based on a target variable and condition.

- **`__correlations(self, df1, df2, name)`**
  - Calculates correlation matrices between two DataFrames and stores them.

- **`do_correlations(self, variables_A, Variables_B, name)`**
  - Creates a correlation matrix for two groups of variables and generates a report.

- **`order_comparisons(self, index)`**
  - Orders comparison results in a standardized format.

- **`generate_corr_report(self, name, select_vars=None)`**
  - Generates a detailed report from a correlation analysis.

- **`chi_sq(self, col, expected_proportions=[0.5, 0.5])`**
  - Performs a chi-squared test on metadata column values to compare observed counts with expected proportions.

- **`plot_differences(self, condition, vars="All", kind="Violin", ylab="", xlab=" ", save=False, show=False)`**
  - Creates violin or box plots to compare variables across different conditions.

- **`plot_correlation(self, var1, var2, save=False, show=False)`**
  - Creates a scatter plot to visualize the correlation between two variables.

## Example Usage

Here’s a brief example to demonstrate how to use the `Stastics` class:

```python
import pandas as pd
import numpy as np
from class_Stastics import Stastics  # Import the class
from sklearn.datasets import load_breast_cancer


# Load an example dataset

dataset = load_breast_cancer(return_X_y=True, as_frame=True)

data = dataset[0]

metadata = dataset[1].to_frame().rename(columns={"target":"Condition"})

metadata["Condition"] = [ "Breast_Cancer" if i==1 else "Control" for i in  metadata.Condition]

# Create an age metadata column randomly
metadata["Age"] = np.astype(np.random.random(size=metadata.shape[0]) * 80,int)

# Create an instance of the Stastics class
statist = Stastics(data, metadata)

# Show the automaticaly generated description
display(statist.adata.uns["Description_All"])

# Filter data based on metadata column (keep the young patients)
ages = statist.adata.obs.Age[ statist.adata.obs.Age < 25 ].tolist()
statist.filter(var_name="Age", subgroup=ages)


# Calculate and generate correlations
correlation_report = statist.do_correlations(
    variables_A=['Age'],
    Variables_B=statist.adata.var_names[:2],
    name="First_10_var"
)

display(correlation_report)


# Compare the condition in the first 10 variables and plot their correlation with Age 
for var in statist.adata.var_names[:2]:
    # Perform comparisons between conditions in each variable
    comparison_results = statist.comparisons_1_1(target=var, condition_name="Condition")

    # Plot correlation with age
    statist.plot_correlation(var1="Age", var2=var, show=True)

# Plot differences in a boxplot
statist.plot_differences(condition="Condition", vars=statist.adata.var_names[:2].tolist(), kind="Box", show=True)

display(comparison_results) # Show the report of the comparisons

```

##  In this example:

-  We initialize the Stastics class with data and metadata from breast_cancer dataset.
-  We show the descriptive statistics, filter data, perform statistical comparisons, and calculate correlations.
-  We then plot differences and correlations using the provided methods.


### **This package is currently under construction.**



