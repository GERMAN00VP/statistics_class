import pandas as pd
import anndata as ad
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import re
import seaborn as sns
import statsmodels.api as sm



class Stastics:

    def __init__(self, data=None, metadata=None,adata=None):

        assert type(adata)!=type(None) or type(data)!=type(None) , "You have to provide some data and metadata (or AnnData) in order to create the object"

        if type(adata)==type(None):

            self.adata = ad.AnnData(data,obs=metadata)
        
        else:

            self.adata = adata

        # Removal of the duplicated columns
        self.adata.obs = self.adata.obs.T.drop_duplicates().T


        # Automatic detection of the column type
        self.adata.obs = self.adata.obs.astype(self.adata.obs.apply(self.__identify_column_type))

        # Create an initial description file
        self.do_description()


        # Initialize a dictionary and a datafame to store continous parways comparisons
        self.dict_comp_1_1 = {"Comparison":["Normal_Data","Test","P-value","Mean_Difference","Hodges_Lehmann_Estimator","N"]}

        self.df_comp_1_1 = pd.DataFrame(self.dict_comp_1_1).T

        # Initialize a dictionary and a datafame to store categorical parways comparisons
        self.dict_chi_sq = {"Name":["Chi2_statistic","P-value","N"]}
        self.adata.uns["CHI_SQ_TABLE"] = pd.DataFrame(self.dict_chi_sq).T

        # Initialize a dictionary for anova results
        self.dict_anova= {"Name":["F","P-value"]}

    def __identify_column_type(self, series):
        """
        Identifies the type of a pandas Series as either categorical or continuous.

        Parameters:
        series (pd.Series): The pandas Series for which the column type is to be identified.

        Returns:
        type: Returns `object` if the series is considered categorical, otherwise `float` for continuous.
        """
        
        # Calculate the number of unique values in the series
        unique_values = len(series.unique())
        
        # Calculate the total number of values in the series
        total_values = len(series)
        
        # Define a heuristic threshold
        unique_ratio = unique_values / total_values
        
        # Threshold to consider as continuous vs categorical
        if unique_ratio < 0.1:
            return object
        else:
            # If any of the non missing values is a string returns an object as well
            if np.any(np.array([re.match(pattern="^-?\d+(\.\d+)?$",string=str(i))==None for i in series.dropna().unique()])):
                return object
            
            return float
        

    def do_description(self, name="All", subset="All"):
        """
        Generates a descriptive statistics summary for the observations in the AnnData object.

        Args:
            name (str, optional): Name to be saved with in the adata.uns object. Defaults to "All".
            subset (tuple, optional): Defaults to "All". 
                If provided, creates a subset of the observations by a column of the metadata and a value for this column. E.g., (Group, control)

        Returns:
            None: The resulting DataFrame is stored in the adata.uns attribute of the object.
        """

        # Access the observation data from the AnnData object
        df = self.adata.obs

        # If a subset is specified, filter the DataFrame
        if subset != "All":
            samples = self.adata.obs[self.adata.obs[subset[0]] == subset[1]].index
            df = self.adata.obs.loc[samples]

        # Initialize lists to store summary statistics
        variables, condicion, cuenta, means, medians, normality,iqr = [], [], [], [], [], [], []

        # Iterate over each column in the DataFrame
        for col in df.columns:
            if df[col].dtype != object:
                # Handle numeric columns

                cuentas = np.nan
                condicion.append(cuentas)
                cuen = len(df[col].dropna()) 
                iqr.append(f"{round(df[col].quantile(0.25),2)} - {round(df[col].quantile(0.75),2)}")
                cuenta.append(f"{cuen} ({round(cuen / len(df) * 100, 2)})")
                variables.append(col)
                mean = df[col].mean()
                desvest = df[col].describe()["std"]
                means.append(f"{round(mean, 2)} ± {round(desvest, 2)}")
                medians.append(round(df[col].median(), 2))
                normality.append(sp.stats.shapiro(df[col].dropna())[1] > 0.05)

            else:
                # Handle categorical columns
        
                cuentas = df[col].value_counts()
                for cuent in cuentas.index:
                    condicion.append(cuent)
                    iqr.append(np.nan)
                    cuen = cuentas[cuent]
                    cuenta.append(f"{cuen} ({round(cuen / len(df) * 100, 2)})")
                    variables.append(col)
                    means.append(np.nan)
                    medians.append(np.nan)
                    normality.append(np.nan)

        # Create a DataFrame from the collected summary statistics
        df_res = pd.DataFrame(
            [variables, condicion, cuenta, means, medians, normality,iqr],
            index=["Variable", "Class", "Count (%)", "Mean ± Std_dev", "Median", "Normal Data","IQR"]
        ).T

        # Store the resulting DataFrame in the adata.uns attribute
        self.adata.uns[f"Description_{name}"] = df_res


    def filter(self,var_name,subgroup="DROPNAS",in_place=True):

        """Method that can filter the adata object by a variable of data or metadata

        Args:
            var_name (str): A column of metadata or data to filter by.

            subgroup (str|list, optional): List of values from the selected variable to be kept . Defaults to "DROPNAS", in that case removes all the obs that contains np.nan values for the selected variable.

        Raises:
            ValueError: If subgroup isnt DROPNAS or a list of values.
            KeyError: If the var_name isnt a column of data or metadata.
        """

        if var_name in self.adata.obs.columns: # Filter by metadata

            if subgroup=="DROPNAS": # Filter out the missing values by var_name

                indexes = self.adata.obs.dropna(subset=var_name).index

                # Keep the non missing values observations for this variable of metadata
                self.adata[indexes,:]

            elif type(subgroup)==list: # Filter out the  values not in subgroup of the var_name column

                indexes= self.adata.obs_names[self.adata.obs[var_name].isin(subgroup)]

            else:

                print(f"Subgroup must be a list or leave empty, not {type(subgroup)}")

                raise ValueError
            


        elif var_name in self.adata.var_names:  # Filter by variables

            if subgroup=="DROPNAS":# Filter out the missing values by var_name

                indexes = self.adata.to_df().dropna(subset=var_name).index

            elif type(subgroup)==list: # Filter out the  values not in subgroup of the var_name column

                indexes= self.adata.obs_names[self.adata.to_df()[var_name].isin(subgroup)]

            else:

                print(f"Subgroup must be a list or leave empty, not {type(subgroup)}")

        else:

            print(f"The var_name '{var_name}' isn`t a column of data or metadata")
            raise KeyError

        if len(indexes)==0:

            print("Any of the observations meet the desired filtering conditions.\nThe filtering won`t be performed.")

        else:

            if in_place:        
                # Do the filtering
                self.adata = self.adata[indexes,:]

                # Regenerate the description with the new data
                self.do_description()

            else:
                # Return a filtered object
                return Stastics(adata = self.adata[indexes,:])
        
        
    def find_var(self, var, adata_df=None):
        """
        Finds and returns the specified variable from the AnnData object.

        Parameters:
        var (str, list, pd.Series, pd.Index, np.ndarray): The variable(s) to be found.
        adata_df (pd.DataFrame, optional): A DataFrame representation of the AnnData object. If None, it will be created.

        Returns:
        pd.Series or pd.DataFrame: The found variable(s) from the AnnData object.

        Raises:
        KeyError: If the specified variable is not found in the AnnData object.
        AssertionError: If the `var` is not of the accepted types.
        """
        
        # Accepted types for the variable
        accepted_types = (list, pd.Series, pd.Index, np.ndarray)  # Use tuple for type checking
        
        # Check if adata_df is not provided or not a DataFrame
        if type(adata_df) != pd.DataFrame:
            # Convert AnnData object to DataFrame if not provided
            adata_df = self.adata.to_df()  # Store DataFrame once
        
        # If var is an accepted type (list, pd.Series, pd.Index, np.ndarray)
        if isinstance(var, accepted_types):
            # Recursively find variables in the list-like structure
            finded_vars = pd.DataFrame([self.find_var(v, adata_df=adata_df) for v in var]).T

            return finded_vars.astype(finded_vars.apply(self.__identify_column_type)) 
        
        # Ensure var is a string
        assert isinstance(var, str), f"The value of var must be str or one of {accepted_types}"
        
        # Check if the variable is in the observation columns
        if var in self.adata.obs.columns:
            variab = self.adata.obs[var]
            return variab.astype(self.__identify_column_type(variab))
        
        # Check if the variable is in the variable names
        if var in self.adata.var_names:
            variab = adata_df[var]
            return variab.astype(self.__identify_column_type(variab))
        
        # Raise an error if the variable is not found
        raise KeyError(f"The variable '{var}' can't be found in the anndata object")


    
    def __check_normality(self ,values:pd.Series,condition=None):

        """ A method that checks the normality of data:

        Args:
            values (pd.Series): the values to check.
            condition (bool, optional): Whether there is a condition (check the normality of both conditions). Defaults to False.


        Returns:
            Boolean, true if normal data.

        """

        if type(condition)!="NoneType": # If a condition is provided

            # Check the normality of every condition
            return np.all([sp.stats.shapiro(values[condition.index[condition==i]].astype(float))[1]>0.05  for i in condition.unique()])
        
        # Check the normality of all values
        return sp.stats.shapiro(values)[1]>0.05
        



    def __hodges_lehmann_estimator(self, group1, group2):
        """
        Calculate the Hodges-Lehmann estimator for two independent samples.
        The Hodges-Lehmann estimator is the median of all pairwise differences 
        between the observations in the two groups.

        Parameters:
        group1 (array-like): First sample
        group2 (array-like): Second sample

        Returns:
        float: Hodges-Lehmann estimator
        """

        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        
        # Compute all pairwise differences
        diffs = []
        for x in group1:
            for y in group2:
                diffs.append(x - y)
        
        # Return the median of differences
        return np.median(diffs)

    
    def comparisons_1_1(self, target, condition_name: str):
        """
        Performs T-test or Mann-Whitney U test comparisons between two variables.

        Args:
            target (str): The target variable to compare.
            condition_name (str): The condition name for grouping the target variable.

        Returns:
            pd.DataFrame: Updated results report of the comparisons.
        """
        
        
        # Extract the values of the target variable, dropping missing values
        df_test = self.find_var(var=[target,condition_name]).dropna()
        df_test = df_test.astype(df_test.apply(self.__identify_column_type))

        values = df_test[target]

        assert values.dtype!= object, "Target must be a continious variable"

        # Extract the corresponding condition values
        condition = df_test[condition_name]

        assert len(condition.unique())==2, "There must be two levels to compare"


        # Check normality of the values based on the condition
        normal = self.__check_normality(values, condition)

        # Split the values into two Series based on the unique conditions
        sep_values = [values[condition.index[condition == cond]].astype(float) for cond in condition.unique()]


        if normal:
            test = "T-test"
            mean_dif = sep_values[0].mean() - sep_values[1].mean()
            median_dif = np.nan
            pval = sp.stats.ttest_ind(sep_values[0], sep_values[1])[1]
        else:
            test = "Mann-Whitney U"
            pval = sp.stats.mannwhitneyu(sep_values[0], sep_values[1])[1]
            mean_dif = np.nan
            median_dif = self.__hodges_lehmann_estimator(sep_values[0], sep_values[1])

        # Store the comparison results in a dictionary
        self.dict_comp_1_1[f"{condition_name}: {target}"] = [normal, test, pval, mean_dif, median_dif, len(values)]

        # Create a DataFrame from the comparison results dictionary
        results_df = pd.DataFrame(self.dict_comp_1_1).set_index("Comparison").T

        if results_df.shape[0] > 1:
            # Perform false discovery rate control
            results_df["FDR"] = sp.stats.false_discovery_control(results_df["P-value"].tolist(), method="bh")
            results_df["Significant"] = results_df["FDR"] < 0.05

        # Update the instance attribute with the results DataFrame
        self.df_comp_1_1 = results_df

        return results_df


    def __correlations(self, df1, df2, name):
        """
        Calculate the correlation matrix between two dataframes and store the results.

        Parameters:
        df1 (pd.DataFrame): First dataframe with the variables of interest.
        df2 (pd.DataFrame): Second dataframe with the variables of interest.
        name (str): Name to prefix the stored results in the annotated data (adata) object.

        Returns:
        None: The function stores the correlation matrix, p-value matrix, and sample size matrix 
            in the `adata.uns` dictionary with keys based on the provided name.
        """
        
        # Initialize the correlation matrix
        corr_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
        pval_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
        n_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
        
        # Calculate the correlations
        for col1 in df1.columns:
            for col2 in df2.columns:

                # Filter non-null values
                valid_idx = df1[col1].notna() & df2[col2].notna()
                x = df1.loc[valid_idx, col1]
                y = df2.loc[valid_idx, col2]

                method = "spearman"

                # Check the conditions required to do Pearson correlation (continuous and normal data)
                if (x.dtype == float) & (y.dtype == float):
                    if (sp.stats.shapiro(x)[1] > 0.05) & (sp.stats.shapiro(y)[1] > 0.05): 
                        method = "pearson"

                # Perform the corresponding correlation
                if method == "spearman": 
                    corr, pval = sp.stats.spearmanr(x, y)
                else:
                    corr, pval = sp.stats.pearsonr(x, y)

                # Store the results in the matrices
                corr_matrix.at[col1, col2] = corr
                pval_matrix.at[col1, col2] = pval
                n_matrix.at[col1, col2] = len(x)

        # Store the matrices in the annotated data (adata) object
        self.adata.uns[f"{name}_Corr"] = corr_matrix
        self.adata.uns[f"{name}_Corr_pval"] = pval_matrix
        self.adata.uns[f"{name}_Corr_N"] = n_matrix



    
    def do_correlations(self, Variables_A, Variables_B, name):
        """
        Uses the __correlation() method from this class to create the correlation matrix 
        from the two groups of variables and generates a report from it.

        Args:
            Variables_A (list): List of variables to correlate from the first group.
            Variables_B (list): List of variables to correlate from the second group.
            name (str): Name to prefix the stored results and generated report.

        Returns:
            pd.DataFrame: DataFrame containing the generated correlation report.
        """

        # Define the dataframes with the selected columns
        df1 = self.find_var(Variables_A)
        if len(df1.shape)==1:
            df1=df1.to_frame()

        df2 = self.find_var(Variables_B)
        if len(df2.shape)==1:
            df2=df2.to_frame()
        
        # Create the correlation matrix
        self.__correlations(df1, df2, name=name)
        
        # Generate the report from the correlation matrix
        results_df = self.generate_corr_report(name=name)

        return results_df




    def order_comparisons(self,index):
        """
        Args:
            index (pd.Index): index of the results DataFrame.

        Returns:
            Values of the index strings oredered.
        """
        parts = index.split(" vs ")
        parts.sort()

        return " vs ".join(parts)


    def generate_corr_report(self, name, select_vars=None):
        """
        Generates a report from a correlation analysis previously run.

        Parameters:
            name (str): The name of the correlation analysis.
            select_vars (list, optional): List of selected variables to include in the report.

        Returns:
            pd.DataFrame: DataFrame containing the correlation report.
        """
        
        matrix_dict = self.adata.uns

        var1, var2, corr, pval, num = [], [], [], [], []

        df_bool = matrix_dict[f"{name}_Corr"] < 0.999  # DataFrame indicating the self-correlation variables

        if select_vars is not None:           
            select_vars = set(select_vars)
            assert len(select_vars.intersection(set(df_bool.columns.tolist() + df_bool.index.tolist()))) == len(select_vars), "One or more variables not found in the correlation matrix"
            
            cols = list(select_vars.intersection(df_bool.columns))
            if len(cols) > 0:
                df_bool = df_bool[cols]
            
            rows = list(select_vars.intersection(df_bool.index))
            if len(rows) > 0:
                df_bool = df_bool.loc[rows]

        for col in df_bool.columns:
            for term in df_bool.index[df_bool[col]].tolist():
                var1.append(col)
                var2.append(term)
                corr.append(matrix_dict[f"{name}_Corr"].loc[term, col])
                pval.append(matrix_dict[f"{name}_Corr_pval"].loc[term, col])
                num.append(matrix_dict[f"{name}_Corr_N"].loc[term, col])

        # Convert the lists to a DataFrame for better visualization
        results_df = pd.DataFrame({
            'Variable_1': var1,
            'Variable_2': var2,
            'Correlation': corr,
            'P-value': pval,
            'N': num
        })

        results_df["Correlated Variables"] = results_df.Variable_1 + " vs " + results_df.Variable_2
        results_df = results_df.dropna().set_index("Correlated Variables")

        # Apply the function to the indexes
        results_df.index = results_df.index.map(self.order_comparisons)

        # Eliminate duplicated indexes
        results_df = results_df[~results_df.index.duplicated(keep='first')]

        # Perform false discovery rate control
        results_df["FDR"] = sp.stats.false_discovery_control(results_df["P-value"], method="bh")
        results_df["Significative"] = results_df["FDR"] < 0.05

        # If select_vars is specified, handle the selection logic
        if select_vars is not None:
            prefix = 1 + max([int(i.split("_")[0]) if i.split("_")[0].isdigit() else 0 for i in self.adata.uns.keys()])
            name = f"{prefix}_Selection"

            # If the report has not been generated yet
            if np.all(np.array([self.adata.uns[result].index.tolist() != results_df.index.tolist() for result in self.adata.uns.keys()])):
                self.adata.uns[f"{name}_Corr_report"] = results_df
        else:
            self.adata.uns[f"{name}_Corr_report"] = results_df

        return results_df
    

    def anova(self,target,condition):
        
        df_test = self.find_var([target,condition]).dropna()
        model= sm.formula.ols(formula=f"{target} ~ {condition}", data = df_test).fit()
        # Realizar ANOVA
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        self.dict_anova[": ".join([target,condition])] = anova_table.loc[condition][["F","PR(>F)"]].tolist()

        # Create a DataFrame from the comparison results dictionary
        results_df = pd.DataFrame(self.dict_anova).set_index("Name").T
        
        # Save the data in adata.uns
        self.adata.uns["ANOVA_results"] = results_df 

        tukey = None
        # If ANOVA is significant, perform Tukey's HSD
        if anova_table['PR(>F)'][0] < 0.05:

            tukey = sm.stats.multicomp.pairwise_tukeyhsd(endog=df_test[target], groups=df_test[condition], alpha=0.05).summary()

        return results_df,tukey


    

    def chi_sq(self, col: str, expected_proportions=[0.5, 0.5]):
        """
        Perform a chi-squared test with the values of a metadata column.

        Parameters:
            col (str): The name of the metadata column to perform the chi-squared test on, or list of var names to compare, 
            the first is the reference.
            expected_proportions (list, optional): The expected proportions for each category. Default is [0.5, 0.5].

        Returns:
            p_value (float): Value P of the chi-squared test.
            None: The results of the chi-squared test are stored in the `adata.uns` dictionary and printed.

        """

        # If two variables are going to be compared
        if type(col)!=str:

            name = "-".join(col)

            chi_df = self.find_var(col).dropna()
            df_contingency = pd.crosstab(chi_df[col[0]],chi_df[col[1]])
            chi2, p_value, dof, expected  = sp.stats.chi2_contingency(df_contingency)
            N = chi_df.shape[0]

        else: 
            name = col
            # Extract the number of observed counts
            data = self.find_var(col).dropna()
            observed_counts = [i for i in data.value_counts()] 
            total_count = sum(observed_counts) 
            # Calculate the expected counts
            expected_counts = [total_count * p for p in expected_proportions] 

            # Perform the chi-squared test
            chi2, p_value = sp.stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

            N = len(data)


        # Store the comparison results in a dictionary
        self.dict_chi_sq[name] =[chi2, p_value, N]

        # Create a DataFrame from the comparison results dictionary
        results_df = pd.DataFrame(self.dict_chi_sq).set_index("Name").T
        
        # Save the data in adata.uns
        self.adata.uns["CHI_SQ_TABLE"] = results_df 

        return results_df


        
    def __get_cov_ellipse(self,cov, nstd):
        """
        Create an ellipse based on a covariance matrix and center point.
        """
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        vx, vy = eigvecs[:,0]
        theta = np.degrees(np.arctan2(vy, vx))
        width, height = 2 * nstd * np.sqrt(eigvals)

        return width, height, theta
    
    def plot_pcoa(self,obsm_key,condition_name,save=False):
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        # Charge the data to plot
        df_plot = self.adata.obsm[obsm_key]

        # Create a column with the condition info
        condition = self.find_var(condition_name)
        df_plot[condition_name] = df_plot.index.map(dict(zip(condition.index, condition)))
        
        centroids = df_plot.groupby(condition_name).mean().reset_index()

        centroids[condition_name] =  "CENTER "+centroids[condition_name]

        sns.scatterplot(ax=axs, data=df_plot, x="PC1", y="PC2", hue=condition_name,s=40)

        # Añadir los centroides al gráfico
        sns.scatterplot(ax=axs, data=centroids, x='PC1', y='PC2', hue=condition_name, marker='s', s=200, legend=False)

        # Obtener el color de las elipses
        palette = sns.color_palette(n_colors=len(df_plot[condition_name].unique()))
        color_dict = {label: palette[idx] for idx, label in enumerate(df_plot[condition_name].unique())}

            # Añadir las elipses
        for label, group in df_plot.groupby(condition_name):
            cov = np.cov(group[['PC1', 'PC2']].T)
            center = group[['PC1', 'PC2']].mean()
            width, height, angle = self.__get_cov_ellipse(cov, 2)
            ellipse = matplotlib.patches.Ellipse(xy=(center["PC1"], center["PC2"]), width=width, 
                            height=height, angle=angle, edgecolor=color_dict[label], 
                            facecolor='none', linestyle='--')
            axs.add_patch(ellipse)

        # Set the texts on the plot
        axs.set_xlabel("PCo1")
        axs.set_ylabel("PCo2")    
        axs.set_title("Principal Coordinates Analysis",fontsize=14,fontweight=800)
        
        plt.legend( bbox_to_anchor=(1, 1),fontsize=9)
        plt.tight_layout()
        if save:
            plt.savefig(save,bbox_inches="tight")

        plt.show()
    


    def plot_differences(self, condition, vars="All", kind="Violin", ylab="", xlab=" ", ylog=False, save=False, show=False):
        """
        Method to create violin or boxplots comparing different states of the data.

        Args:
            condition (str): The condition to compare the variables by.
            vars (str, optional): Name of the variables to compare (list, np.array, or pd.Series). Defaults to "All".
            kind (str, optional): "Box" for boxplot, "Violin" for violin plot. Defaults to "Violin".
            ylab (str, optional): Label of the y-axis. Defaults to "".
            xlab (str, optional): The name of the variables, appears in the x label. Defaults to " ".
            save (bool, optional): Path to save the figure. Defaults to False.
            show (bool, optional): Whether to show the plot or close it directly. Defaults to False.

        Raises:
            AttributeError: If an unsupported kind of plot is requested.
        """

        if isinstance(vars, str) and vars == "All":  # If default, all variables are selected
            vars = self.adata.var_names
            df_var = self.find_var(var=vars)
        
        elif isinstance(vars, str):  # If a single variable is selected
            df_var = self.find_var(var=vars).to_frame()

        else:  # User selects some of the variables
            df_var = self.find_var(var=vars)
            
        df_var[condition] = df_var.index.map(dict(zip(self.adata.obs.index, self.adata.obs[condition])))


        df_var = df_var.melt(id_vars=condition, value_name="value", var_name=xlab.upper())

        # Create the plot
        plt.figure(figsize=(10, 6))

        if kind == "Violin":
            sns.violinplot(x=xlab, y="value", data=df_var, inner='quart', hue=condition, split=False)

        elif kind == "Box":
            sns.boxplot(x=xlab, y="value", data=df_var, hue=condition, fliersize=0)
            # Enhance aesthetics
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add a subtle grid
        
        else:
            print("Option not implemented yet")
            raise AttributeError
        
        ax = sns.stripplot(x=xlab, y="value", data=df_var, hue=condition, dodge=True, jitter=True, palette='dark:k', alpha=0.5, legend=False)


        if ylog:
            plt.yscale("log")

        # Add labels
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xticks(rotation=25, ha='right')

        # If only one variable, do not show the ticks
        if len(df_var[xlab].unique()) == 1:
            plt.xticks([])

        # Place the legend
        plt.legend(title=condition, loc='best')

        sns.despine()

        
        # Extraer los tick labels actuales del eje x y generar unos nuevos con el valor de la n
        x_tick_labels = [f"{item.get_text().capitalize()} (n={df_var.dropna()[" "].value_counts()[item.get_text()]}) " for item in ax.get_xticklabels()]

        # Aplicar los nuevos tick labels al eje x
        ax.set_xticklabels(x_tick_labels)


        if save:
            plt.savefig(save, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


    def plot_correlation(self, var1, var2, save=False, show=False):
        """
        Plot the correlation between two variables using a scatter plot.

        Args:
            var1 (str): The name of the first variable.
            var2 (str): The name of the second variable.
            save (bool or str, optional): Path to save the figure. Defaults to False.
            show (bool, optional): Whether to show the plot or close it directly. Defaults to False.

        Returns:
            None
        """

        df_plot = self.find_var([var1, var2]).dropna()

        # Create the plot
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(x=var1, y=var2, data=df_plot)
        
        # Add a subtle grid
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        sns.despine()

        plt.title(f"Correlation: {var2} vs {var1} (n={df_plot.shape[0]})")

        if save:
            plt.savefig(save, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

