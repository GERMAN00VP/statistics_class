import pandas as pd
import anndata as ad
import numpy as np
import scipy as sp




class Stastics:

    def __init__(self, data, metadata):

        self.adata = ad.AnnData(data,obs=metadata)

        self.do_description()
    
    def do_description(self,name="All"):
        """_summary_

        Args:
            name (str, optional): Name to be saved with in the adata.uns object. Defaults to "All".
        """

        df = self.adata.obs

        #for col in df_description_all_bg.columns:
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




    # Método para calcular la matriz de correlaciones
    def __correlations(self,df1,df2,name,method='spearman'):

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
        
        self.adata.varm[f"{name}_Corr"] = corr_matrix
        self.adata.varm[f"{name}_Corr_pval"] = pval_matrix
        self.adata.varm[f"{name}_Corr_N"] = n_matrix
    
    def correlations_metadata(self):

        df1 = self.adata.to_df()
        df2 = self.adata.obs

        self.__correlations(df1,df2,name="Variables_Metadata")

    def correlations_variables(self):

        df1 = self.adata.to_df()
        df2 = self.adata.to_df()

        self.__correlations(df1,df2,name="Variables_Variables")

    def generate_corr_report(self,name="Variables_Variables"):

        """
        Description: This method generates a report from a correlation analysis previously run.

        Parameters:
            name: (str) The name of the tcorrelation analysis.


        Returns:
            report: (pd.DataFrame)
        """


        correlated_vars,corr,pval,num = [], [], [], []

        df_bool = self.adata.varm[f"{name}_Corr"]<0.999  # A dataframe that indicates the self correlation variables

        


        for col in df_bool.columns:

            for term in df_bool.index[df_bool[col]].tolist():

                correlated_vars.append(f"{col} vs {term}")
                corr.append(self.adata.varm[f"{name}_Corr"].loc[term,col])
                pval.append(self.adata.varm[f"{name}_Corr_pval"].loc[term,col])
                num.append(self.adata.varm[f"{name}_Corr_N"].loc[term,col])


        # Convert the lists to a DataFrame for better visualization
        results_df = pd.DataFrame({
            'Correlated Variables': correlated_vars,
            'Correlation': corr,
            'P-value': pval,
            'N': num
        })

        results_df=results_df.sort_values(by="Correlation").dropna().drop_duplicates(subset="Correlation").set_index("Correlated Variables")

        results_df["FDR"] = sp.stats.false_discovery_control(results_df["P-value"],method = "bh")

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