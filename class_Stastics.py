import pandas as pd
import anndata as ad
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

class Stastics:

    def __init__(self, data, metadata):

        self.adata = ad.AnnData(data,obs=metadata)

        # Removal of the duplicated columns
        self.adata.obs = self.adata.obs.T.drop_duplicates().T


        # Automatic detection of the column type
        self.adata.obs = self.adata.obs.astype(self.adata.obs.apply(self.identify_column_type))

        # Create an initial description file
        self.do_description()


        self.dict_comp_1_1 = {"Comparisson":["Normal_Data","Test","P-value","Mean_Difference","Hodges_Lehmann_Estimator"]}

        self.df_comp_1_1 = pd.DataFrame(self.dict_comp_1_1).T

    def filter(self,var_name,subgroup="DROPNAS"):

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
            # Do the filtering
            self.adata = self.adata[indexes,:]

            # Regenerate the description with the new data
            self.do_description()



    def identify_column_type(self,series):
        unique_values = len(series.unique())
        total_values = len(series)
        
        # Definir un umbral heurístico
        unique_ratio = unique_values / total_values
    
        # Umbral para considerar como continuo vs categórico
        if unique_ratio < 0.1:
            return object
        
        else:
            return float
        
    def find_var(self,var):

        if var in self.adata.obs.columns:
            
            variable = self.adata.obs[var]


        elif var in self.adata.var_names:

            variable = self.adata.to_df()[var]

        else:
            print(f"The variable '{var} can`t be found in the anndata object" )
            raise KeyError
        
        return variable


    
    def do_description(self,name="All",subset="All"):

        """_summary_

        Args:
            name (str, optional): Name to be saved with in the adata.uns object. Defaults to "All".
            subset (tuple, optional): Defaults to "All". 
                If provided, creates a subset of the observations by a column fo the metadata and a value for this column.Ej: (Group, control)

        """

        df = self.adata.obs

        if subset!= "All":

            samples = self.adata.obs[self.adata.obs[subset[0]]==subset[1]].index

            df = self.adata.obs.loc[samples]

        

        variables,condicion,cuenta,means,medians = [], [], [], [], []

        for col in df.columns:

            if df[col].dtype!=object:
                cuentas=np.nan
                condicion.append(cuentas)

                cuen = len(df[col]-df[col].isna().sum())

                cuenta.append(f"{cuen} ({round(cuen/len(df)*100,2)})")

                variables.append(col)

                mean = df[col].mean()
                desvest = df[col].describe()["std"]
                means.append(f"{round(mean,2)} ± {round(desvest,2)}")
                medians.append(round(df[col].median(),2))


            else:

                cuentas = df[col].value_counts()
                for cuent in cuentas.index:
                    condicion.append(cuent)

                    cuen = cuentas[cuent]

                    cuenta.append(f"{cuen} ({round(cuen/len(df)*100,2)})")
                    variables.append(col)
                    means.append(np.nan)
                    medians.append(np.nan)

        df_res= pd.DataFrame([variables,condicion,cuenta,means,medians],index=["Variable","Class","Count (%)","Media ± Desv_est","Mediana"]).T

        self.adata.uns[f"Description_{name}"]= df_res


    
    def __check_normality(self ,values:pd.Series,condition=None):

        """ A method that checks the normality of data:

        Args:
            values (pd.Series): the values to check.
            condition (bool, optional): Whether there is a condition (check the normality of both conditions). Defaults to False.

    

        Returns:
            Boolean, true if normal data.

        """

        if type(condition)!="NoneType":

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

    
    def comparisons_1_1(self,target,condition_name:str):
        """Do the t student or Mann Whitney Comparisons between twoo variables

        Args:
            target (_type_): _description_
            condition_name (str): _description_

        Returns:
            _type_: _description_
        """

        df = self.adata.to_df()

        values = df[target].dropna() # Extract the values (dropping missing values)

        condition = self.adata.obs[condition_name][values.index]

        normal= self.__check_normality(values,condition)

        # Values splitted in two Series by condition

        sep_values= [values[condition.index[condition==cond]].astype(float) for cond in condition.unique()]

        if normal:

            test= "T-test"

            mean_dif = sep_values[0].mean() - sep_values[1].mean()

            median_dif = np.nan

            pval = sp.stats.ttest_ind(sep_values[0],sep_values[1])[1]
        
        else:

            test= "Mann-Whitney U"

            pval =  sp.stats.mannwhitneyu(sep_values[0],sep_values[1])[1]

            mean_dif = np.nan

            median_dif = self.__hodges_lehmann_estimator(sep_values[0], sep_values[1])
        
        self.dict_comp_1_1[f"{condition_name}: {target}"]= [normal,test,pval,mean_dif,median_dif]

        self.df_comp_1_1= pd.DataFrame(self.dict_comp_1_1).set_index("Comparisson").T

        return self.df_comp_1_1


    # Método para calcular la matriz de correlaciones
    def __correlations(self,df1,df2,name,method='spearman',save_in="varm"):

        # Inicializar la matriz de correlaciones
        corr_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
        pval_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
        n_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns)
        
        # Calcular las correlaciones
        for col1 in df1.columns:
            for col2 in df2.columns:

                # Extract the common non missing value indexes
                intersection_indexes = list(set(df1[col1].dropna().index).intersection(set(df2[col2].dropna().index)))

                # Run the correlation
                if method == 'spearman':
                    corr, pval = sp.stats.spearmanr(df1[col1].loc[intersection_indexes], df2[col2].loc[intersection_indexes])
                elif method == 'pearson':
                    corr, pval = sp.stats.pearsonr(df1[col1].loc[intersection_indexes], df2[col2].loc[intersection_indexes])
                else:
                    raise ValueError("Method not supported. Use 'spearman' or 'pearson'.")
                

                corr_matrix.loc[col1, col2] = corr
                pval_matrix.loc[col1, col2] = pval
                n_matrix.loc[col1, col2]= len(intersection_indexes)

        if save_in=="varm":
        
            self.adata.varm[f"{name}_Corr"] = corr_matrix
            self.adata.varm[f"{name}_Corr_pval"] = pval_matrix
            self.adata.varm[f"{name}_Corr_N"] = n_matrix

        elif save_in=="uns":
            self.adata.uns[f"{name}_Corr"] = corr_matrix
            self.adata.uns[f"{name}_Corr_pval"] = pval_matrix
            self.adata.uns[f"{name}_Corr_N"] = n_matrix

        else:
            print("This is not a suitable place for storing the data")
    

    
    def correlations_metadata(self,save_in="uns"):

        df1 = self.adata.obs
        df2 = self.adata.obs

        self.__correlations(df1,df2,name="Metadata_Metadata",save_in=save_in)

    
    def correlations_metadata_variables(self):

        df1 = self.adata.to_df()
        df2 = self.adata.obs

        self.__correlations(df1,df2,name="Variables_Metadata")


    def correlations_variables(self):

        df1 = self.adata.to_df()
        df2 = self.adata.to_df()

        self.__correlations(df1,df2,name="Variables_Variables")

    def order_comparisons(self,index):
        """
        Args:
            index (pd.Index): index of the results DataFrame

        Returns:
            Values of the index strings oredered
        """
        parts = index.split(" vs ")
        parts.sort()
        return " vs ".join(parts)


    def generate_corr_report(self,name="Variables_Metadata",save_in="varm",select_vars = None):

        """
        Description: This method generates a report from a correlation analysis previously run.

        Parameters:
            name: (str) The name of the tcorrelation analysis.


        Returns:
            report: (pd.DataFrame)
        """

        if save_in=="varm":

            matrix_dict = self.adata.varm

        else: 
            matrix_dict = self.adata.uns
        


        var1,var2,corr,pval,num = [], [], [], [], []

        df_bool = matrix_dict[f"{name}_Corr"]<0.999  # A dataframe that indicates the self correlation variables

        if select_vars!=None:

            okay = False

            select_vars = set(select_vars)

            cols = list(select_vars.intersection(df_bool.columns))

            if len(cols)>0:
                df_bool=df_bool[cols]
                okay = True
            
            rows = list(select_vars.intersection(df_bool.index))

            if len(rows)>0:
                df_bool=df_bool.loc[rows]
                okay = True

            assert okay, "The variables selected doesn`t belong to this correlation matrix"


        for col in df_bool.columns:

            for term in df_bool.index[df_bool[col]].tolist():

                var1.append(col)
                var2.append(term)
                corr.append(matrix_dict[f"{name}_Corr"].loc[term,col])
                pval.append(matrix_dict[f"{name}_Corr_pval"].loc[term,col])
                num.append(matrix_dict[f"{name}_Corr_N"].loc[term,col])


        # Convert the lists to a DataFrame for better visualization
        results_df = pd.DataFrame({
            'Variable_1': var1,
            'Variable_2': var2,
            'Correlation': corr,
            'P-value': pval,
            'N': num
        })

        results_df["Correlated Variables"]= results_df.Variable_1+" vs "+results_df.Variable_2

        results_df=results_df.dropna().set_index("Correlated Variables")

        # Apply the function to the indexes
        results_df.index = results_df.index.map(self.order_comparisons)

        # Eliminate duplicated indexes
        results_df = results_df[~results_df.index.duplicated(keep='first')]

        # Do the false discover rate control
        results_df["FDR"] = sp.stats.false_discovery_control(results_df["P-value"],method = "bh")

        results_df["Significative"] = results_df["FDR"]<0.05

        # In case we selected 
        if select_vars!=None:
             
            prefix = 1+ max([int(i.split("_")[0]) if i.split("_")[0].isdigit() else 0 for i in self.adata.uns.keys()])
            
            name = f"{prefix}_Selection"

            # If the report hasnt been generated yet
        
            if np.all(np.array([self.adata.uns[result].index.tolist() !=results_df.index.tolist() for result in self.adata.uns.keys()])):

                self.adata.uns[f"{name}_Corr_report"] = results_df
                
        else:
            self.adata.uns[f"{name}_Corr_report"] = results_df          

        

        return results_df
    
    
    def chi_sq(self, col:str, expected_proportions = [0.5, 0.5]):

        """
        Perform a chi-squared test with the values of a metadata column

        
        """
        observed_counts = [i for i in self.adata.obs[col].value_counts()] # Extract the  number of observed counts
        total_count = sum(observed_counts) 
        expected_counts = [total_count * p for p in expected_proportions] # Calculate the expected counts

        # Realizar la prueba de chi-cuadrado
        chi2, p_value = sp.stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

        self.adata.uns[f"CHI_SQ_{col}"] = {"P-val":p_value,"Stastic":chi2}

        if p_value < 0.05:
            print(f"La proporción de cuentas de las variables de la columna {col} se desvía significativamente del lo esperado.")
        else:
            print(f"La proporción de cuentas de las variables de la columna {col} no se desvía significativamente del lo esperado.")



    def plot_differences(self, condition ,varsname = " ",vars = "All", kind = "Violin",save=False,ylab = "",show=False):
        """Method to do violin or boxplots comparing different states of the data

        Args:
            condition (str): The condition to compare the variables by.
            varsname (str, optional): The name of the variables, appars in the x label. Defaults to " ".
            vars (str, optional): Name of the variables to compare (list, np.array or pd.Series). Defaults to "All".
            kind (str, optional): Box for boxplot. Defaults to "Violin".
            save (bool, optional): path to save the figure. Defaults to False.
            ylab (str, optional): label of the y axe. Defaults to "".
            show (bool, optional): whether to show the plot or close it directly.

        Raises:
            AttributeError: If someone requests an other kind of plot
        """

        
        if isinstance(vars, str) and vars == "All": # In case of defoult all variables are selected
            vars = self.adata.var_names
            df_var = self.adata.to_df()[vars]
        
        elif isinstance(vars, str): # In case a single variable is selected
            df_var = self.adata.to_df()[vars].to_frame()

        else: # The user selects some of the variables
            df_var = self.adata.to_df()[vars]
            
        df_var[condition] = df_var.index.map(dict(zip(self.adata.obs.index,self.adata.obs[condition])))
        df_var = df_var.melt(id_vars=condition,value_name="value",var_name=varsname)

        
        # Crear el plot
        plt.figure(figsize=(10, 6))

        if kind=="Violin":
            sns.violinplot(x=varsname, y= "value", data=df_var, inner='quart', hue=condition, split=False)

        elif kind=="Box":
            sns.boxplot(x=varsname, y= "value", data=df_var,  hue=condition,fliersize=0)
            # Mejorar la estética
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Añadir una cuadrícula sutil
        
        else:
            print("Option not implemetnted yet")
            raise AttributeError
        
        sns.stripplot(x=varsname, y="value", data=df_var, hue=condition, dodge=True, jitter=True, color='k', alpha=0.5,legend=False)


        # Añadir títulos y etiquetas
        plt.xlabel(varsname)
        plt.ylabel(ylab)
        plt.xticks(rotation=25, ha='right')

        # Si solo hay una variable no mostrar los ticks
        if len(df_var[varsname].unique())==1:
            plt.xticks([])


        # Colocar la leyenda
        plt.legend(title=condition,loc='best')

        
        sns.despine()

        if save:
            plt.savefig(save,bbox_inches="tight")

        if show:
            plt.show()

        else:
            plt.close()  


    def plot_correlation(self, var1 ,var2, save=False,show=False):
        
        variable1 = self.find_var(var1)
        variable2 = self.find_var(var2)

        df_plot = pd.DataFrame([variable1,variable2]).T

        # Crear el plot
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(x=var1, y=var2, data=df_plot)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Añadir una cuadrícula sutil


        sns.despine()

        if save:
            plt.savefig(save,bbox_inches="tight")

        if show:
            plt.show()

        else:
            plt.close()       

