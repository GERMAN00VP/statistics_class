# MicrobiomePy

**MicrobiomePy** is currently under development and originates as a branch of the `Statistics_class` package. 

The goal is to create a package that builds upon the foundation of the aforementioned package but focuses exclusively on 16S metagenomics data. From the outset, the package will feature a syntax and structure tailored specifically for analyzing this type of data. Additionally, it will adopt a new modular organization, initially comprising four main modules:

## **Modules**

### 1. `MicrobiomePy.basics`
This module will provide the basic structure for data storage and normalization. It will also include essential utility functions, such as:

- Filtering cases and variables.
- Searching for variables.
- Generating reports.

The design will aim to optimize functionality and improve cohesion. For example, a unified reporting model will be implemented to ensure consistency across all statistical results, regardless of the model or type of comparison.

---

### 2. `MicrobiomePy.stats`
This module will focus on functions related to basic statistical analysis for any variable within the object. It will also support customizable descriptive and analytical reports, including:

- Metadata variables.
- Statistical metrics (e.g., mean, median, interquartile range).
- Relevant comparisons (e.g., Chi-square, odds ratio, Fisher’s exact test, t-tests, Mann-Whitney U, correlations).

---

### 3. `MicrobiomePy.plotting`
This module will include a variety of predefined plots. The aim is to strike a balance between utility and flexibility, always returning plots in a way that allows users to modify specific details easily. This approach minimizes the need for excessive parameter input. Planned plots include:

- Boxplots.
- Violin plots.
- Volcano plots.
- PCoA plots.

Additional plots will be added based on user needs and feedback.

---

### 4. `MicrobiomePy.analyst`
This module will house functions specifically for metagenomic analysis, such as:

- Diversity analysis.
- Taxonomic abundance comparisons.
- Functional analysis (potentially).

---

## **Third-Party Integration**
Most of these functions will leverage existing third-party packages. The aim of MicrobiomePy is to act as a wrapper that streamlines and unifies the analysis process, providing a user-friendly interface for metagenomics data analysis.

---

This modular design ensures that **MicrobiomePy** remains flexible, cohesive, and tailored to the specific needs of 16S metagenomics data analysis.
