import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import scipy.stats as stats
from io import BytesIO
import xlsxwriter  
from datetime import datetime
import zipfile
from scipy.stats import skew, kurtosis


# =============================
# App Configuration
# =============================
st.set_page_config(page_title="Heart Disease EDA", layout="wide")
st.title("❤️ Heart Disease EDA Dashboard")

# =============================
# Sidebar Navigation
# =============================
section = st.sidebar.radio("Navigate to:", [
    "Upload Dataset",
    "Unit I: EDA Overview",
    "Unit II: Data Selection & Transformation",
    "Unit III: Univariate Analysis",
    "Unit IV: Bivariate Analysis",
    "Unit V: Multivariate & Time Series",
    "Download Processed Data"
])

# =============================
# File Upload
# =============================
@st.cache_data
def load_data(file=None):
    if file:
        return pd.read_csv(file)
    return pd.read_csv("C:/Users/DELL/OneDrive/Desktop/EDA project/Dataset/Heart_disease_cleveland_new.csv")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = load_data(uploaded_file)

# =============================
# Data Preprocessing
# =============================
impute_strategy = st.sidebar.selectbox("Missing Value Strategy", ["Mean", "Median", "Drop Rows"])

if impute_strategy == "Mean":
    df.fillna(df.mean(numeric_only=True), inplace=True)
elif impute_strategy == "Median":
    df.fillna(df.median(numeric_only=True), inplace=True)
elif impute_strategy == "Drop Rows":
    df.dropna(inplace=True)

duplicates = df.duplicated()
if duplicates.any():
    df = df[~duplicates]

numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["gender", "cp", "slope", "thal", "fbs", "restecg", "exang"]

# Normalization
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))
df_transformed = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)

# =============================
# SECTION: Dataset Upload
# =============================
if section == "Upload Dataset":
    st.subheader("Original Dataset Preview")
    st.write(df.head())

# =============================
# UNIT I: EDA Overview
# =============================
elif section == "Unit I: EDA Overview":
    st.subheader("📊 Dataset Overview")
    
    # Basic Dataset Information
    st.markdown(f"**Shape of Dataset:** {df.shape}")
    st.markdown("**Columns in Dataset:**")
    st.write(list(df.columns))

    # Data Types of Each Column
    st.markdown("### Data Types of Each Column:")
    st.write(df.dtypes)

    # Summary statistics
    st.subheader("📈 Dataset Summary Statistics")
    st.dataframe(df.describe())
    
    with st.expander("ℹ️ What does this table show?"):
        st.markdown("""
        This table provides descriptive statistics for numeric columns:
        - **Count**: Number of non-null entries.
        - **Mean**: Average value.
        - **Std**: Standard deviation.
        - **Min/Max**: Range of data.
        - **25%, 50%, 75%**: Quartiles.
        - **Skew**: Measure of asymmetry. (Right skew for positive skew, left for negative.)
        - **Kurtosis**: Measure of tails (how heavy they are). High kurtosis means more extreme outliers.
        """)

    # Select a numeric feature from the dataset
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_col = st.selectbox("Select Numeric Column for Analysis", num_cols)

    # Summary for the selected column
    st.markdown(f"### 📊 Selected Feature: **{selected_col}**")
    skewness = df[selected_col].skew()
    kurtosis_value = df[selected_col].kurtosis()

    st.write(f"**Skewness**: {skewness:.2f}")
    st.write(f"**Kurtosis**: {kurtosis_value:.2f}")

    # Boxplot for the selected feature
    st.subheader(f"📦 Boxplot and Outlier Detection of {selected_col}")
    fig_box, ax_box = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[selected_col], ax=ax_box, color="#ff7f0e")
    ax_box.set_title(f"Boxplot: {selected_col}", fontsize=12)
    st.pyplot(fig_box)

    # Calculate IQR and Outliers
    Q1 = df[selected_col].quantile(0.25)
    Q3 = df[selected_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[selected_col][(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
    st.write(f"Outliers detected in {selected_col}:")
    st.write(outliers)

    # KDE Plot for the selected feature
    st.subheader(f"📈 KDE Plot of {selected_col}")
    fig_kde, ax_kde = plt.subplots(figsize=(6, 4))
    sns.kdeplot(df[selected_col], ax=ax_kde, shade=True, color='skyblue')
    ax_kde.set_title(f"KDE Plot of {selected_col}", fontsize=12)
    st.pyplot(fig_kde)

    # Skewness and Kurtosis from the KDE plot (same as before)
    data = df[selected_col].dropna()
    skewness_val = skew(data)  # Compute skewness
    kurt_value = kurtosis(data)  # Compute kurtosis, renamed to avoid conflict

    st.write(f"Real-time **Skewness**: {skewness_val:.2f}")
    st.write(f"Real-time **Kurtosis**: {kurt_value:.2f}")  # Use the renamed variable

    with st.expander("ℹ️ What do these plots show?"):
        st.markdown(f"""
        - **Boxplot**: 
            - A **boxplot** shows the distribution of data based on the **interquartile range (IQR)**.
            - **IQR** is calculated as the difference between the 75th percentile (**Q3**) and the 25th percentile (**Q1**).
            - Outliers are detected using the formula:  
              - **Lower Bound** = **Q1** - 1.5 * IQR  
              - **Upper Bound** = **Q3** + 1.5 * IQR
            - Any data points outside of the whiskers are considered outliers.
            - **Key Statistics from Boxplot**:
              - **Q1 (25th percentile)**: {Q1:.2f}
              - **Q3 (75th percentile)**: {Q3:.2f}
              - **IQR (Interquartile Range)**: {IQR:.2f}
              - **Lower Bound**: {lower_bound:.2f}
              - **Upper Bound**: {upper_bound:.2f}

        - **KDE Plot**: 
            - A **KDE plot** is a smoothed version of the histogram, showing the estimated probability density function of the data.
            - **Skewness** of the data indicates the direction of the tail: 
              - A **positive skew** means the right tail is longer, while a **negative skew** means the left tail is longer.
            - **Kurtosis** tells us about the "tailedness" of the distribution:
              - High kurtosis indicates heavy tails (more extreme values).
              - Low kurtosis indicates light tails (fewer extreme values).
            - The **KDE** visually shows how the data points are distributed, highlighting the density of observations in different ranges.
            - **Real-time KDE Analysis**:
              - **Skewness**: {skewness_val:.2f}
              - **Kurtosis**: {kurt_value:.2f}
        """)

elif section == "Unit II: Data Selection & Transformation":
    st.subheader("🔍 Data Selection & Transformation")

    # ------------------------------
    # Missing Values Check
    # ------------------------------
    st.markdown("### 🔹 Missing Values Check")
    missing_values = df.isnull().sum()
    st.write(missing_values)

  

    if st.button("Explain Missing Values"):
        st.info("This section highlights missing values in your dataset. "
                "Columns with high missing values might need imputation or removal before analysis.")

    # ------------------------------
    # Basic Row/Column Selection
    # ------------------------------
    st.markdown("### 🔹 Row & Column Selection")
    st.write(df.loc[0:5, ["age", "chol", "target"]])

    # Slider for Age Filter
    st.markdown("### 🔹 Filter Rows by Age")
    age_threshold = st.slider("Select minimum age (standardized)", min_value=float(df["age"].min()), 
                              max_value=float(df["age"].max()), value=0.5)
    filtered_df = df[df["age"] > age_threshold]
    st.write(f"Rows where Age > {age_threshold:.2f}")
    st.write(filtered_df)

    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df["chol"], kde=True, ax=ax1)
    ax1.set_title("Cholesterol Distribution (Filtered by Age)")
    st.pyplot(fig1)

    if st.button("Explain Age Filter Chart"):
        st.info("This histogram shows how cholesterol values are distributed for patients older than the selected standardized age. "
                "It helps identify if higher age groups also tend to have higher cholesterol.")

    # ------------------------------
    # GroupBy + Visualization
    # ------------------------------
    st.markdown("### 🔹 Group By Chest Pain Type")
    grouped_cp = df.groupby("cp")["chol"].mean()
    st.write(grouped_cp)

    fig2, ax2 = plt.subplots()
    grouped_cp.plot(kind="bar", ax=ax2, color="skyblue")
    ax2.set_ylabel("Mean Cholesterol")
    ax2.set_title("Mean Cholesterol by Chest Pain Type")
    st.pyplot(fig2)

    if st.button("Explain CP Grouping"):
        st.info("The bar plot shows how average cholesterol varies by chest pain type. "
                "This can indicate if certain types are associated with worse lipid profiles.")

    # ------------------------------
    # Pivot Table
    # ------------------------------
    st.markdown("### 🔹 Pivot Table: Chol by Gender & Target")
    pivot = df.pivot_table(values="chol", index="gender", columns="target", aggfunc="mean")
    st.write(pivot)

    fig3, ax3 = plt.subplots()
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax3)
    ax3.set_title("Pivot Heatmap: Mean Cholesterol by Gender & Target")
    st.pyplot(fig3)

    if st.button("Explain Pivot Heatmap"):
        st.info("This heatmap shows how mean cholesterol levels differ by gender and heart disease status. "
                "You can see whether gender is a contributing factor in cholesterol levels for heart patients.")

    # ------------------------------
    # Value Counts Bar Chart (Optional)
    # ------------------------------
    st.markdown("### 🔹 Categorical Value Distribution")
    cat_col_sel = st.selectbox("Select a categorical column to view distribution", categorical_features)
    val_counts = df[cat_col_sel].value_counts()

    fig4, ax4 = plt.subplots()
    sns.barplot(x=val_counts.index, y=val_counts.values, ax=ax4)
    ax4.set_ylabel("Count")
    ax4.set_title(f"Distribution of {cat_col_sel}")
    st.pyplot(fig4)

    if st.button("Explain Category Distribution"):
        st.info(f"This chart shows how frequently each category appears in the `{cat_col_sel}` column. "
                f"It helps detect class imbalance and skewed feature distributions.")

    # ------------------------------
    # Optional: Crosstab Table
    # ------------------------------
    st.markdown("### 🔹 Crosstab Analysis")
    crosstab_x = st.selectbox("X-axis (categorical)", categorical_features, key="crosstab_x")
    crosstab_y = st.selectbox("Y-axis (categorical)", categorical_features, key="crosstab_y")
    crosstab_table = pd.crosstab(df[crosstab_x], df[crosstab_y])
    st.dataframe(crosstab_table)

    fig5, ax5 = plt.subplots()
    sns.heatmap(crosstab_table, annot=True, cmap="magma", fmt="d", ax=ax5)
    ax5.set_title(f"Crosstab: {crosstab_x} vs {crosstab_y}")
    st.pyplot(fig5)

    if st.button("Explain Crosstab"):
        st.info(
            f"This heatmap compares the distribution of two categorical features: **{crosstab_x}** and **{crosstab_y}**. "
            f"It helps identify relationships or interaction patterns in categorical features."
        )
# =============================
# UNIT III: Univariate Analysis
# =============================
elif section == "Unit III: Univariate Analysis":
    st.subheader("Numerical Column Analysis")
    col = st.selectbox("Select a numerical column", numerical_features)

    # Expanded Statistical Summary
    stats_summary = {
        "Mean": df[col].mean(),
        "Median": df[col].median(),
        "Mode": df[col].mode()[0],
        "Variance": df[col].var(),
        "Std Dev": df[col].std(),
        "Skewness": df[col].skew(),
        "Kurtosis": df[col].kurt(),
        "Min": df[col].min(),
        "Max": df[col].max(),
        "Range": df[col].max() - df[col].min(),
        "Interquartile Range (IQR)": df[col].quantile(0.75) - df[col].quantile(0.25),
        "Sum": df[col].sum(),
        "Count": df[col].count(),
        "25th Percentile": df[col].quantile(0.25),
        "75th Percentile": df[col].quantile(0.75)
    }
    st.write(stats_summary)

    if st.button("Explain Statistical Summary"):
        st.info("This summary provides insights into the selected numerical column. "
                "Mean, median, and mode indicate central tendency, while variance and standard deviation "
                "measure the spread of data. Skewness helps understand symmetry, and kurtosis shows the peak shape.")

    # Histogram with KDE
    fig1, ax1 = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax1, bins=30)
    ax1.set_title(f"Histogram of {col} with KDE")
    st.pyplot(fig1)

    if st.button("Explain Histogram"):
        st.info("This histogram visualizes the distribution of the selected numerical column. "
                "KDE (Kernel Density Estimation) adds a smooth curve to show the probability density.")

    # Boxplot for Outlier Detection
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df[col], ax=ax2)
    ax2.set_title(f"Boxplot of {col}")
    st.pyplot(fig2)

    if st.button("Explain Boxplot"):
        st.info("The boxplot helps detect outliers and understand variability. The median is inside the box, "
                "and whiskers indicate the spread of the data.")

    # Violin Plot for Distribution Insight
    fig3, ax3 = plt.subplots()
    sns.violinplot(x=df[col], ax=ax3)
    ax3.set_title(f"Violin Plot of {col}")
    st.pyplot(fig3)

    if st.button("Explain Violin Plot"):
        st.info("A violin plot combines boxplots and KDE curves, showing both the distribution and variability.")

    # Swarm Plot for Data Distribution Density
    fig4, ax4 = plt.subplots()
    sns.swarmplot(x=df[col], ax=ax4)
    ax4.set_title(f"Swarm Plot of {col}")
    st.pyplot(fig4)

    if st.button("Explain Swarm Plot"):
        st.info("The swarm plot helps visualize individual data points to detect clustering and spread.")

    # Scaling & Standardization
    st.subheader("Scaling & Standardizing")
    scaled_data = (df[col] - df[col].min()) / (df[col].max() - df[col].min())  # Min-Max Scaling
    standardized_data = (df[col] - df[col].mean()) / df[col].std()  # Z-score Standardization

    st.write("Scaled Data (Min-Max Normalization):")
    st.write(scaled_data.head())

    st.write("Standardized Data (Z-score):")
    st.write(standardized_data.head())

    if st.button("Explain Scaling & Standardization"):
        st.info("Scaling (Min-Max) transforms values between 0 and 1, ensuring a comparable range. "
                "Standardization (Z-score) converts data to have mean 0 and standard deviation 1.")

    st.subheader("Categorical Column Analysis")
    cat_col = st.selectbox("Select a categorical column", categorical_features + ["target"])

    # Value Counts with Percentage Display
    st.write("Frequency Distribution:")
    value_counts = df[cat_col].value_counts()
    value_percent = df[cat_col].value_counts(normalize=True) * 100
    freq_df = pd.DataFrame({'Count': value_counts, 'Percentage': value_percent})
    st.write(freq_df)

    if st.button("Explain Frequency Distribution"):
        st.info("This table displays the occurrence of different categories in the selected column, "
                "helping identify common values and possible imbalances.")

    # Bar Chart for Categorical Feature
    fig5, ax5 = plt.subplots()
    sns.countplot(x=cat_col, data=df, ax=ax5)
    ax5.set_title(f"Distribution of {cat_col}")
    st.pyplot(fig5)

    if st.button("Explain Bar Chart"):
        st.info("The bar chart visualizes frequency distributions of categorical values.")

    # Pie Chart Representation
    fig6, ax6 = plt.subplots()
    df[cat_col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax6)
    ax6.set_title(f"Pie Chart of {cat_col}")
    st.pyplot(fig6)

    if st.button("Explain Pie Chart"):
        st.info("The pie chart shows the proportion of different categories in the selected feature.")

    # Crosstab Analysis (Optional)
    st.subheader("Crosstab Analysis")
    crosstab_x = st.selectbox("X-axis (categorical)", categorical_features, key="crosstab_x")
    crosstab_y = st.selectbox("Y-axis (categorical)", categorical_features, key="crosstab_y")
    crosstab_table = pd.crosstab(df[crosstab_x], df[crosstab_y])
    st.dataframe(crosstab_table)

    fig7, ax7 = plt.subplots()
    sns.heatmap(crosstab_table, annot=True, cmap="magma", fmt="d", ax=ax7)
    ax7.set_title(f"Crosstab: {crosstab_x} vs {crosstab_y}")
    st.pyplot(fig7)

    if st.button("Explain Crosstab"):
        st.info("The crosstab shows relationships between two categorical variables. The heatmap visually highlights "
                "patterns and interactions in the data.")


elif section == "Unit IV: Bivariate Analysis":
    st.subheader("Bivariate Analysis")

    # ================================
    # Scatter Plot with Regression Line
    # ================================
    st.markdown("### Scatter Plot with Regression")
    x = st.selectbox("X-axis", numerical_features, key="scatter_x")
    y = st.selectbox("Y-axis", [col for col in numerical_features if col != x], key="scatter_y")
    fig, ax = plt.subplots()
    sns.regplot(x=df[x], y=df[y], ax=ax, line_kws={"color": "red"})
    ax.set_title(f"{y} vs {x}")
    st.pyplot(fig)

    if st.button("Explain Scatter Plot"):
        st.info(
            f"This scatter plot shows the relationship between **{x}** and **{y}** with a regression line. "
            f"A visible slope suggests a linear trend. Outliers or clustering may suggest deeper interactions."
        )

    # ================================
    # Pearson Correlation + Heatmap
    # ================================
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    if st.button("Explain Correlation Heatmap"):
        st.info(
            "The heatmap displays pairwise Pearson correlation coefficients. "
            "Values near ±1 imply strong linear relationships. This helps identify multicollinearity and variable relationships."
        )

    # ================================
    # Percentage Tables for Categorical Data
    # ================================
    st.markdown("### Percentage Tables")
    cat_var = st.selectbox("Select a categorical feature for percentage analysis", categorical_features)
    percentage_table = df[cat_var].value_counts(normalize=True) * 100
    st.write(percentage_table)

    if st.button(f"Explain Percentage Table for {cat_var}"):
        st.info(
            f"The percentage table shows the relative frequency of values in **{cat_var}** as percentages. "
            f"This helps identify class distributions and imbalances."
        )

    # ================================
    # Analyzing Contingency Tables (Chi-Square)
    # ================================
    st.markdown("### Chi-Square Test (Categorical vs Target)")
    table = pd.crosstab(df[cat_var], df["target"])
    chi2, p, dof, expected = stats.chi2_contingency(table)

    st.write("Contingency Table:")
    st.dataframe(table)
    st.write(f"Chi² Statistic = {chi2:.3f}")
    st.write(f"Degrees of Freedom = {dof}")
    st.write(f"P-value = {p:.4f}")

    # Decision based on p-value
    if p < 0.05:
        st.success("✅ Result: Reject H₀ — Significant association detected.")
    else:
        st.warning("❌ Result: Fail to reject H₀ — No significant association found.")

    fig, ax = plt.subplots()
    sns.heatmap(table, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
    ax.set_title(f"Heatmap of {cat_var} vs Target")
    st.pyplot(fig)

    if st.button(f"Explain Chi-Square for {cat_var}"):
        st.info(
            f"The chi-square test evaluates whether **{cat_var}** and the target are independent. "
            f"If the p-value < 0.05, we reject H₀, suggesting a significant relationship."
        )

    # ================================
    # Handling Several Batches - ANOVA
    # ================================
    st.markdown("### ANOVA Test (Multiple Groups)")
    anova_col = st.selectbox("Choose a numerical feature for ANOVA", numerical_features)
    anova_groups = [df[df["target"] == i][anova_col] for i in df["target"].unique()]
    f_stat, p_value_anova = stats.f_oneway(*anova_groups)

    st.write(f"F-statistic = {f_stat:.3f}")
    st.write(f"P-value = {p_value_anova:.4f}")

    if p_value_anova < 0.05:
        st.success(f"✅ Result: Reject H₀ — Statistically significant difference in {anova_col} across multiple target groups.")
    else:
        st.warning(f"❌ Result: Fail to reject H₀ — No significant difference in {anova_col} across groups.")

    if st.button(f"Explain ANOVA for {anova_col}"):
        st.info(
            f"ANOVA assesses whether there are significant differences in the mean of **{anova_col}** "
            f"across multiple categories in the **target** variable."
        )

    # ================================
    # Scatterplots & Resistant Lines (Robust Regression)
    # ================================
   # Define x and y before using them
    # Assign unique keys to X and Y select boxes
    x = st.selectbox("Select X-axis", df.select_dtypes(include=['number']).columns, key="scatter_x_unique")
    y = st.selectbox("Select Y-axis", [col for col in df.select_dtypes(include=['number']).columns if col != x], key="scatter_y_unique")

    st.markdown("### Resistant Line (Robust Regression)")

    fig = sns.lmplot(x=x, y=y, data=df)  # lmplot directly generates a figure
    st.pyplot(fig)  # Render it in Streamlit

    if st.button(f"Explain Resistant Line for {y} vs {x}"):
        st.info(
            "Resistant regression lines minimize the influence of outliers, making them more robust "
            "than traditional regression models in scatterplots."
    )
    # ================================
    # Transformations (Log & Square Root)
    # ================================
    st.markdown("### Variable Transformations")
    transform_col = st.selectbox("Select a numerical column for transformations", numerical_features)

    # Apply transformations
    df["log_transformed"] = np.log1p(df[transform_col])  # Log transformation
    df["sqrt_transformed"] = np.sqrt(df[transform_col])  # Square root transformation

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["log_transformed"], kde=True, ax=ax[0])
    ax[0].set_title(f"Log Transformation of {transform_col}")

    sns.histplot(df["sqrt_transformed"], kde=True, ax=ax[1])
    ax[1].set_title(f"Square Root Transformation of {transform_col}")

    st.pyplot(fig)

    if st.button(f"Explain Transformations for {transform_col}"):
        st.info(
            "Log and square root transformations help stabilize variance and normalize distributions. "
            "These techniques are useful when dealing with skewed data."
        )

    # ================================
    # T-Test: Mean Difference by Target
    # ================================
    st.markdown("### Independent Samples T-Test")
    ttest_col = st.selectbox("Choose a numerical feature for t-test", numerical_features, key="ttest")
    group0 = df[df["target"] == 0][ttest_col]
    group1 = df[df["target"] == 1][ttest_col]
    t_stat, p_val = stats.ttest_ind(group0, group1)

    fig, ax = plt.subplots()
    sns.boxplot(x="target", y=ttest_col, data=df, ax=ax)
    ax.set_title(f"{ttest_col} Distribution by Target")
    st.pyplot(fig)

    st.write(f"T-statistic = {t_stat:.3f}")
    st.write(f"P-value = {p_val:.4f}")

    if p_val < 0.05:
        st.success(f"✅ Result: Reject H₀ — Statistically significant difference in {ttest_col}.")
    else:
        st.warning(f"❌ Result: Fail to reject H₀ — No significant difference in {ttest_col}.")

    if st.button(f"Explain T-Test for {ttest_col}"):
        st.info(
            f"This test compares the mean of **{ttest_col}** between two groups (target = 0 and 1). "
            f"If the p-value < 0.05, we reject the null hypothesis, indicating a significant difference."
        )


# =============================
# UNIT V: Multivariate + Time Series
# =============================
elif section == "Unit V: Multivariate & Time Series":
    st.subheader("Multivariate Analysis")

    # ================================
    # Pairplot with Target Hue
    # ================================
    st.markdown("### Pairplot with Target Hue")
    fig = sns.pairplot(df[numerical_features + ["target"]], hue="target")
    st.pyplot(fig)
    if st.button("Explain Pairplot"):
        st.info(
            "This pairplot visualizes all numerical features pairwise. "
            "It helps detect relationships and separability of target classes (0 = no heart disease, 1 = heart disease). "
            "You can see how some variables (like thalach or oldpeak) cluster differently depending on the target."
        )

    # ================================
    # Three-Variable Contingency Table
    # ================================
    st.markdown("### Three-Variable Contingency Table")
    cat1 = st.selectbox("Select first categorical feature", categorical_features, key="contingency1")
    cat2 = st.selectbox("Select second categorical feature", categorical_features, key="contingency2")
    cat3 = st.selectbox("Select third categorical feature", categorical_features, key="contingency3")

    contingency_table = pd.crosstab([df[cat1], df[cat2]], df[cat3])
    st.dataframe(contingency_table)

    if st.button("Explain Contingency Table"):
        st.info(
            f"This contingency table shows the relationship among **{cat1}, {cat2}, and {cat3}**. "
            "Analyzing three variables together provides deeper insights into category interactions."
        )

    # ================================
    # Violin Plot
    # ================================
    st.markdown("### Violin Plot")
    selected_violin_col = st.selectbox("Choose a variable for Violin Plot", numerical_features)
    fig, ax = plt.subplots()
    sns.violinplot(x="target", y=selected_violin_col, data=df, ax=ax)
    ax.set_title(f"Violin plot of {selected_violin_col} by Target")
    st.pyplot(fig)
    if st.button(f"Explain Violin Plot for {selected_violin_col}"):
        st.info(
            f"This violin plot shows the distribution of **{selected_violin_col}** for each heart disease group (target). "
            f"If one group has a different shape, median, or spread, it means this variable may help distinguish between the classes."
        )

    # ================================
    # KDE Plot by Target
    # ================================
    st.markdown("### KDE Plot by Target")
    selected_kde_col = st.selectbox("Choose a variable for KDE Plot", numerical_features, key="kde")
    fig, ax = plt.subplots()
    for t in df["target"].unique():
        sns.kdeplot(df[df["target"] == t][selected_kde_col], label=f"Target {t}", ax=ax)
    ax.set_title(f"KDE Plot of {selected_kde_col} by Target")
    ax.legend()
    st.pyplot(fig)
    if st.button(f"Explain KDE Plot for {selected_kde_col}"):
        st.info(
            f"This KDE plot compares the distribution of **{selected_kde_col}** between heart disease (target=1) and no disease (target=0). "
            f"A large difference in curves suggests that {selected_kde_col} is a strong predictor."
        )

    # ================================
    # Causal Explanations with Regression
    # ================================
    st.markdown("### Causal Explanation: Linear Regression")
    causal_x = st.selectbox("Select Independent Variable", numerical_features, key="causal_x")
    causal_y = st.selectbox("Select Dependent Variable", numerical_features, key="causal_y")

    fig, ax = plt.subplots()
    sns.regplot(x=df[causal_x], y=df[causal_y], ax=ax, line_kws={"color": "red"})
    ax.set_title(f"Causal Relationship: {causal_y} vs {causal_x}")
    st.pyplot(fig)

    if st.button(f"Explain Causal Relationship for {causal_y} vs {causal_x}"):
        st.info(
            f"This regression plot shows the relationship between **{causal_x}** and **{causal_y}**. "
            "A strong trend suggests that changes in **{causal_x}** may cause variations in **{causal_y}**."
        )

    # ================================
    # Clustered Heatmap
    # ================================
    st.markdown("### Clustered Heatmap")
    fig = sns.clustermap(df[numerical_features].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig.fig)

    if st.button("Explain Clustered Heatmap"):
        st.info(
            "This clustered heatmap groups numerical variables by correlation. "
            "High correlation (close to 1 or -1) means those features are strongly related. "
            "Use this to detect redundant variables that may be removed or interpreted together."
        )

    # ================================
    # Time-Series Analysis
    # ================================
    st.subheader("Time-Series Analysis")

    # Time-Based Indexing
    df_time = df.copy()
    df_time["date"] = pd.date_range("2023-01-01", periods=len(df))
    df_time.set_index("date", inplace=True)

    # Select Aggregation Method
    aggregation_method = st.selectbox("Choose aggregation method", ["Mean", "Sum"], key="ts_agg")

    # Resampling Interval
    resample_interval = st.selectbox("Choose time interval", ["D", "W", "M"], key="ts_resample")

    # Apply Resampling
    if aggregation_method == "Mean":
        resampled = df_time.resample(resample_interval).mean()
    else:
        resampled = df_time.resample(resample_interval).sum()

    # Select Variable for Time Series
    line_col = st.selectbox("Choose Variable to Plot", numerical_features, key="ts_variable")

    # Display Time-Series Chart
    st.line_chart(resampled[line_col])

    if st.button(f"Explain Time Series of {line_col}"):
        st.info(
            f"This time-series plot shows how **{line_col}** changes over different intervals (Daily, Weekly, Monthly). "
            f"Aggregation method ({aggregation_method}) determines whether averages or total values are displayed."
        )

# =============================
# Download Button
# =============================
elif section == "Download Processed Data":
    st.subheader("📤 Download Encoded + Scaled Data")

    # File format selection
    file_format = st.selectbox(
        "Select file format",
        ["CSV", "Excel"],
        help="Choose the format you want to download the data in."
    )

    # Encoding selection (for CSV only)
    encoding = None
    if file_format == "CSV":
        encoding = st.selectbox("Select CSV encoding", ["utf-8", "utf-16", "ISO-8859-1"], index=0)

    # Date in filename option
    use_date = st.checkbox("Append current date to filename", value=True)

    # Custom filename input
    base_filename = st.text_input("Enter filename (without extension)", value="transformed_data")

    # Row limit selection
    row_option = st.radio("Select rows to download", ["All Rows", "Top 100", "Top 500"])
    if row_option == "Top 100":
        data_to_download = df_transformed.head(100)
    elif row_option == "Top 500":
        data_to_download = df_transformed.head(500)
    else:
        data_to_download = df_transformed

    # Column selection
    selected_columns = st.multiselect("Select columns to include", options=df_transformed.columns.tolist(), default=df_transformed.columns.tolist())
    data_to_download = data_to_download[selected_columns]

    # Add timestamp to filename if selected
    filename = base_filename
    if use_date:
        filename += "_" + datetime.now().strftime("%Y%m%d")

    # Excel sheet name input (if Excel selected)
    sheet_name = "Sheet1"
    if file_format == "Excel":
        sheet_name = st.text_input("Enter Excel sheet name", value="Sheet1")

    # Preview
    with st.expander("🔍 Preview Transformed Data"):
        st.dataframe(data_to_download.head(10))

    # Create buffer
    buffer = BytesIO()

    # Write file content
    if file_format == "CSV":
        data_to_download.to_csv(buffer, index=False, encoding=encoding)
        mime = "text/csv"
        file_ext = ".csv"
    else:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            data_to_download.to_excel(writer, index=False, sheet_name=sheet_name)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        file_ext = ".xlsx"

    buffer.seek(0)
    file_data = buffer.getvalue()

    # Show file size
    file_size_kb = round(len(file_data) / 1024, 2)
    st.info(f"📦 Estimated file size: {file_size_kb} KB")

    # Optional ZIP compression
    compress = st.checkbox("Compress file to ZIP")
    if compress:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(filename + file_ext, file_data)
        zip_buffer.seek(0)
        download_data = zip_buffer
        download_filename = filename + ".zip"
        mime = "application/zip"
    else:
        download_data = buffer
        download_filename = filename + file_ext

    # Download button
    if st.download_button(
        label="📥 Download File",
        data=download_data,
        file_name=download_filename,
        mime=mime,
    ):
        st.success("✅ Download initiated successfully!")

# =============================
# Footer
# =============================
st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit")
