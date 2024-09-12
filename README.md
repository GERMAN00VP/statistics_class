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
import matplotlib.pyplot as plt

# Load an example dataset

dataset = load_breast_cancer(return_X_y=True, as_frame=True)

data = dataset[0]

metadata = dataset[1].to_frame().rename(columns={"target":"Condition"})

# Set a seed 
np.random.seed(seed=1)

# Create an age metadata column randomly but including a bias based in Condition
metadata["Age"] =10+(np.random.random(size=metadata.shape[0]) * 80).astype(float) \
    + np.array([ 0.2 if i==1 else 0.1 for i in  metadata.Condition]) \
    * (np.random.random(size=metadata.shape[0]) * 80).astype(float)*metadata["Condition"]

metadata["Condition"] = [ "Breast Cancer" if i==1 else "Control" for i in  metadata.Condition]

# Create a categorical variable with 3 levels randomly assigned
metadata["Alimentación"] = np.random.randint(0, high=3,size=metadata.shape[0]) 

# Create an instance of the Stastics class
statist = Stastics(data, metadata)

# Show the automaticaly generated description
display(statist.adata.uns["Description_All"])

# Show a description of the Breast cancer patients metadata

statist.do_description(name="Breast_Cancer",subset=("Condition","Breast Cancer"))
display(statist.adata.uns["Description_Breast_Cancer"])

# Show the PCA by the condition
statist.plot_pca(vars=statist.adata.var_names,hue="Condition",title="Principal Component Analysis of Breast Cancer Dataset")

# Test for correlations with age
cor_results = statist.do_correlations(Variables_A="Age",Variables_B=statist.adata.var_names,name="Age_correlations",return_df=True).sort_values(by="P-value")
# Normal correlation
statist.plot_correlation(var_y=cor_results.loc[cor_results.Significative,"Variable_1"][0]
                         ,var_x="Age",theme="notebook")
# Correlation by condition
statist.plot_correlation(var_y="mean radius",var_x="Age",hue="Condition",palette=["red","blue"],theme="paper")
cor_results.head()


# Filter data based on metadata column (keep older patients)
ages = statist.adata.obs.Age[ statist.adata.obs.Age > 45 ].tolist()
statist.filter(var_name="Age", subgroup=ages)

# Compare the condition in the first 8 variables and plot their correlation with Age 
for var in statist.adata.var_names[:8]:

    # Perform comparisons between conditions in each variable
    comparison_results = statist.comparisons_1_1(target=var, condition="Condition")

    # Perform an anova analysis for the condition with 3 levels
    anova_res = statist.anova(target=var,condition="Alimentación",name="ANOVA_ALIMENTACIÓN")

comparison_results.sort_values(by="P-value",inplace=True) # Sotr the results by significance

# Plot differences by condition
statist.plot_differences(condition="Condition", # Choose the variable to split the data by
                         vars= comparison_results.index.str.replace("Condition: ?","",regex=True)[:3] # Show the 3 most significatly different variables
                         ,palette=["salmon","skyblue"], # Manualy set the colours of the plot
                         kind="Box", # Kind of plot, other options are Violin and Bar
                         ylog=True, # Log scale in the y axis
                         theme="paper", # Set a sns theme for the plot
                         tick_label_names=["Radius","Concave Points","Area"], # Change the x tick label names
                         show_n=False # Dont show the number of samples plotted
                         ) 

# Plot differences with the defaut options except for show and kind
_,ax = statist.plot_differences(condition="Condition", vars=statist.adata.var_names[:3].tolist(),
                                kind= "Bar",
                                show="Edit" # Return the axes of the figure
                                )
# You can also choose to return the plot in case you want to add some changes
plt.legend(title="Condición",loc="upper left")

# Change the transparency of the data points
for collection in ax.collections:
    collection.set_alpha(0.1) 
plt.show()
display(comparison_results) # Show the report of the comparisons


# Show the anova results, as we created this condition randomly we didnt expect to get any significant results
anova_res  = anova_res[0].sort_values(by="P-value")
display(anova_res)
# Plot the three most significant results 
statist.plot_differences(vars=anova_res.iloc[:3,:].index.str.replace(":.*","",regex=1),
                         condition="Alimentación",ylog=1)

 # Test the dependency of the variables (Chi squared test) and show the overlap with a venn plot
statist.plot_venn(["Alimentación","Condition"],figsize=(20,10))
statist.chi_sq(col=["Alimentación","Condition"])
```

##  In this example:

-  We initialize the Stastics class with data and metadata from breast_cancer dataset.
-  We show the descriptive statistics, filter data, perform statistical comparisons, and calculate correlations.
-  We then plot differences and correlations using the provided methods.


### **This package is currently under construction.**



