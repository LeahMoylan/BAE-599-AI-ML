import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Streamlit Page Setup ---
st.set_page_config(page_title="BAE 599 Midterm Project: Students' Social Media Addiction", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📊 BAE 599 Midterm Project")
st.write("""
### Project Summary
(Created after completion of all steps)
         
**Dataset:**  "Students' Social Media Addiction" from Kaggle

**Goal:**  Identify which features most strongly predict whether a student is at high risk for social media addiction.

**Model:**  Logistic Regression

**Results:**  

The most powerful predictor of social media addiction risk out of the features studied is number of conflicts over social media, followed by average daily usage hours. 
         
The most powerful protective feature against social media addiction out of those studied is a higher mental health score.


GitHub Repository:  https://github.com/LeahMoylan/BAE-599-AI-ML/tree/main
         
""")

# --- Step 1: Load Data ---
with st.expander("Step 1. Dataset selection & inspection"):
    st.markdown("""
    #### Description & Citation
    I chose the dataset "Students' Social Media Addiction" from Kaggle (https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships). The dataset contains survey responses from high school through graduate students who use social media. Descriptions of each feature collected are below:


    | **Feature Name**                 | **Type**        | **Description** |
    |----------------------------------|-----------------|-----------------|
    | Student_ID                       | Integer         | Unique respondent identifier |
    | Age                              | Integer         | Age in years |
    | Gender                           | Categorical     | Male / Female |
    | Academic_Level                   | Categorical     | High School / Undergraduate / Graduate |
    | Country                          | Categorical     | Country of residence |
    | Avg_Daily_Usage_Hours            | Float           | Average hours per day spent on social media |
    | Most_Used_Platform               | Categorical     | Instagram / Facebook / TikTok / etc. |
    | Affects_Academic_Performance     | Boolean         | Self-reported impact on academics (Yes/No) |
    | Sleep_Hours_Per_Night            | Float           | Average nightly sleep hours |
    | Mental_Health_Score              | Integer         | Self-rated mental health (1 = poor, 10 = excellent) |
    | Relationship_Status              | Categorical     | Single / In Relationship / Complicated |
    | Conflicts_Over_Social_Media      | Integer         | Number of relationship conflicts due to social media |
    | Addicted_Score                   | Integer         | Self-reported social media addiction (1 = low, 10 = high) |


    This is a topic of personal interest to me. I want to explore what factors influence social media addiction. 
    A preview of the dataframe is shown below.""")

    @st.cache_data
    def load_data():
        df = pd.read_csv("Students Social Media Addiction.csv")
        return df

    df = load_data()
    st.write("#### Dataset Preview")
    st.write(f"**Shape of the dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(11))


    # --- Step 2: Descriptive Statistics ---
    st.markdown("#### Data Inspection and Visualizations")
    st.markdown("""
    To get a initial idea of relationships between variables, we are going to use various inspection techniques like histograms and box plots.
    This will provide an overview of the central tendency, dispersion, and shape of the data.
    """)

    st.dataframe(df.describe().T)


    # Numeric columns for plotting
    numeric_cols = [
        'Age',
        'Avg_Daily_Usage_Hours',
        'Sleep_Hours_Per_Night',
        'Mental_Health_Score',
        'Conflicts_Over_Social_Media',
        'Addicted_Score'
    ]

    # 🔹 Identify Outliers Using IQR Method
    st.markdown("#### Potential Outliers Detected (IQR Method)")
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
        st.write(f"**{col}:** {len(outliers)} outliers")

    # 🔹 Create Histograms (Distribution) and Box Plots (Outliers)
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(14, 25))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    fig.suptitle('Distributions, Outliers, and Skewness of Numeric Variables', fontsize=16, y=0.9)

    for i, col in enumerate(numeric_cols):
        # Calculate skewness
        skew_val = df[col].skew()

        # Histogram (Distribution)
        sns.histplot(df[col], kde=False, ax=axes[i, 0], color='skyblue')
        axes[i, 0].set_title(f'Distribution of {col}', fontsize=12)
        axes[i, 0].text(
            0.95, 0.95, f"Skewness: {skew_val:.2f}",
            transform=axes[i, 0].transAxes,
            ha='right', va='top',
            fontsize=10, color='black', weight='bold'
        )
        axes[i, 0].text(
            0.5, -0.25, f"Outliers: {outlier_counts[col]}",
            transform=axes[i, 0].transAxes,
            ha='center', va='top',
            fontsize=12, color='red'
        )

        # Box Plot
        sns.boxplot(x=df[col], ax=axes[i, 1], color='lightcoral')
        axes[i, 1].set_title(f'Box Plot of {col}', fontsize=12)
        axes[i, 1].set_ylabel('')
        axes[i, 1].text(
            0.5, -0.25, f"Outliers: {outlier_counts[col]}",
            transform=axes[i, 1].transAxes,
            ha='center', va='top',
            fontsize=12, color='red'
        )

    st.pyplot(fig)

    # 🔹 Correlation Heatmap
    st.markdown("#### Correlation Heatmap of Key Numeric Variables")
    fig_corr, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black', ax=ax)
    ax.set_title('Correlation Heatmap of Key Numeric Variables', fontsize=16)
    st.pyplot(fig_corr)

    # Categorical columns
    categorical_cols = [
        'Gender',
        'Academic_Level',
        'Most_Used_Platform',
        'Affects_Academic_Performance',
        'Relationship_Status'
    ]


    # 🔹 Plot Categorical Distributions (excluding Country)
    st.markdown("### Distributions of Categorical Variables")
    fig_cat, axes = plt.subplots(nrows=len(categorical_cols), ncols=1, figsize=(10, 20))
    plt.subplots_adjust(hspace=0.7)
    fig_cat.suptitle('Distributions of Categorical Variables', fontsize=16, y=0.92)

    for i, col in enumerate(categorical_cols):
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis', ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].set_xlabel('Count', fontsize=12)
        axes[i].set_ylabel(col, fontsize=12)

    st.pyplot(fig_cat)

    st.markdown("""
    #### Histograms & Boxplots

    The  visualizations show that all distributions of numerical features are only slightly skewed and close to normally distributed (slightly skewed = skewness of ±0.5-0/1). Using IQR, the only feature with outliers is Avg_Daily_Usage_Hours with 3 outliers. Ordinal features (i.e. Age, Mental_Health_Score, Conflicts_Over_Social_Media, and Addicted_Score) had numeric labels, so we also visualized them with histograms and boxplots. 

    #### Heatmap

    The heatmap shows strong positive correlations between addiction and usage hours and between addiction and conflict. There are strong negative correlations between mental health and addiction and between mental health and conflict.

    #### Bar charts

    The bar charts show counts of categorical variables. Gender is pretty evenly split. Academic level is heavy on the graduate and undergraduate categories. There is a lot of variety in most used platform. Academic performance is a good mix, and relationship status leans towards single or in a relationship rather than complicated. Country was not shown in a bar chart because there was a very large number of categories. Instead, percentages of the top 10 countries were displayed in the output with Indian, US, and Canada being the most frequent.""")





# --- Step 4: Feature Engineering ---
with st.expander("Step 2. Datasat cleaning & preparation"):
    st.markdown("""
    #### Removing Irrelevant Columns
    Student_ID is a unique identifier (not predictive). Country has high cardinality and is dropped to prevent excessive sparse features.
    """)

    cols_to_drop = ["Student_ID", "Country"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors="ignore")
    print(f"Shape after dropping irrelevant columns: {df.shape}")
    st.write(f"Dropped columns: {cols_to_drop}")

    st.markdown("""
    #### Handling Outliers 
    From Step 1 above, we saw that only Avg_Daily_Usage_Hours contained outliers when capping with 1.5*IQR, and that only three outliers were found. Although a few extreme values were identified, they were retained because they likely represent genuine high-use cases rather than errors. Keeping these observations preserves the natural variability of the data and allows the model to learn from both typical and extreme patterns of behavior.
    """)

    st.markdown("""
    #### Encoding Categorical Variables
    Gender and Academic performance are binary features. Turning categorical data into numeric data is necessary for most machine learning models. We will use label encoding which is efficient for binary features, and avoids multicollinearity from One-Hot Encoding.
    """)

    le = LabelEncoder()
    binary_cols = ["Gender", "Affects_Academic_Performance"]
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    st.write(f"Label encoded: {binary_cols}")

    st.markdown("""
    #### Grouping and One-Hot Encoding

    Grouping: 

    Most_Used_Platform has high cardinality. Grouping the categories to a manageable set (Top 5 + 'Other'), reducing sparsity and model complexity.  
                
    Groups:
    1. Instagram  
                
    2. TikTok  
            
    3. Facebook  

    4. Twitter  
                
    5. Other (all remaining platforms)
                

                
    One-Hot Encoding: 

    Converts nominal categories into numerical form without imposing a false ordinal relationship.
    """)

    top_n = 5
    top_platforms = df["Most_Used_Platform"].value_counts().nlargest(top_n).index
    df["Most_Used_Platform_Grouped"] = np.where(
        df["Most_Used_Platform"].isin(top_platforms),
        df["Most_Used_Platform"],
        "Other"
    )
    df = df.drop(columns=["Most_Used_Platform"])

    nominal_cols = ["Academic_Level", "Relationship_Status", "Most_Used_Platform_Grouped"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    st.write(f"One-Hot encoded: {nominal_cols}")

    st.markdown("""
    #### Scaling Numeric Features
    Scaling ensures all features contribute equally to the model by placing them on the same scale (mean=0, std=1), which is vital for distance-based and gradient-based ML algorithms.
    """)

    numeric_to_scale = [
        'Age',
        'Avg_Daily_Usage_Hours',
        'Sleep_Hours_Per_Night',
        'Mental_Health_Score',
        'Conflicts_Over_Social_Media',
        'Addicted_Score'
    ]

    scaler = StandardScaler()
    df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])

    print(f"Standard Scaled: {numeric_to_scale}")
    print(f"Final Shape: {df.shape}\n")
    print("Data Preprocessing Complete. Final Data Info:")
    df.info()


# --- Step 5: Model Setup ---
with st.expander("Step 3. Model Selection"):
    st.markdown("""
    #### Model Choice: Logistic Regression

    Logistic Regression is the most appropriate initial model due to three key factors identified during the data inspection:

    A. Alignment with Observed Data Structure

    * Strong Linear Relationships: The initial inspection's correlation heatmap and pair plots showed strong, clear linear relationships between Addicted_Score and key predictors like Avg_Daily_Usage_Hours (positive) and Sleep_Hours_Per_Night (negative).

    * Logistic Regression is a Linear Model: It assumes that the log-odds of the outcome (the probability of being high-risk) can be modeled as a linear combination of the input features. This assumption aligns perfectly with the strong linear trends observed in the data.


    B. Interpretability

    * Research Requirement: For a research assignment, justification and insight are paramount. Logistic Regression is an inherently interpretable model.

    * Actionable Insights: The coefficients of the fitted model can be transformed into odds ratios. For example, you can state: "A one-unit increase in Avg_Daily_Usage_Hours makes a student X% more likely to be in the high addiction risk group." This is a powerful, justifiable, and easy-to-explain result for your final presentation.


    C. Data Preparation Match

    * Clean and Scaled Input: The data is now fully cleaned, scaled (StandardScaler), and encoded (OHE/LabelEncoder). Logistic Regression performs optimally with scaled numeric features because scaling prevents features with larger magnitudes (like Conflicts\_Over\_Social\_Media) from unfairly dominating the coefficient fitting process.

                
    D. Comparison to Other Models
                
    * Logistic Regression was chosen over complex models like Random Forests and Support Vector Machines because it provides direct interpretability through coefficients and odds ratios, allowing each factor’s influence on addiction risk to be clearly understood.

    * Unlike tree-based or nonlinear models, it avoids becoming a “black box,” making results transparent and explainable — essential when studying human behavior.

    * Logistic Regression also performs efficiently with smaller datasets and fewer features, where deep or ensemble models may overfit or add unnecessary complexity.

    * While models like SVMs might capture subtle nonlinearities, Logistic Regression’s simplicity, stability, and interpretability make it better suited for this dataset’s educational and analytical goals.
                
    """)

with st.expander("""## Step 4. Model application and training"""):

    st.markdown("""
    #### Choose a target variable
    Logistic regression is a binary classification, meaning we are predicting the probability of an outcome that has only two possible values. Therefore, we need to select a target variable. Since this dataset focuses on social media addiction, we chose Addicted_Score as the target. It is not binary, so we need to convert it to be so. We can convert this feature by considering >= the 75th percentile as the definition for "High Risk". This will provide us two possible values: "High Risk" or "Low Risk."

    The code converts Addicted_Score to binary and identifies it as the target variable. Then, the code establishes a 70/30 train/test split.""")

    addiction_threshold = df['Addicted_Score'].quantile(0.75)
    df['High_Addiction_Risk'] = np.where(df['Addicted_Score'] >= addiction_threshold, 1, 0)
    df = df.drop(columns=['Addicted_Score'])

    X = df.drop('High_Addiction_Risk', axis=1)
    y = df['High_Addiction_Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    st.write(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    st.markdown("""
    #### Apply and Tune the Model
    
    The code applies the model using sklearn's LogisticRegression. Then, the model’s performance is improved through hyperparameter tuning.

    The hyperparameter C controls the regularization strength — how much the model penalizes large coefficients to avoid overfitting.

    Low C values = stronger regularization (simpler model, less overfitting).

    High C values = weaker regularization (more complex model, higher variance).

    A grid search is performed to tune the regularization parameter C. 
    This process uses 5-fold cross-validation to test multiple regularization levels and find the optimal value.
    """)


    log_reg = LogisticRegression(solver='liblinear', random_state=42)
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(log_reg, param_grid, scoring='accuracy', cv=5, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    st.write(f"Optimal **C** value found: {grid_search.best_params_['C']}")
    st.write(f"Best cross-validation score: **{grid_search.best_score_:.4f}**")

    st.markdown("""
    The grid search results show that the optimal C value was 1.0, meaning the model performed best with a moderate amount of regularization. This indicates that the model’s performance does not require overly tight constraints to generalize well.

    The best cross-validation score of 0.9959 (99.59%) indicates exceptionally strong performance during training and validation.
    This value represents the average accuracy achieved across five separate folds of the training data. Such a consistently high score means the model was able to correctly classify most observations regardless of how the data was split.
    
    """)

    st.markdown("""
    #### Run the model on the test set

    Now, we will evaluate the model on the held-out test set to assess how well it performs on new data. 
    Accuracy, precision, recall, F1-score, and confusion matrix are used to assess performance.
                
    Results are shown below. A discussion of each result is provided in Step 5.
                

    """)




    best_log_reg = grid_search.best_estimator_
    y_pred_tuned = best_log_reg.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_tuned)
    st.write(f"**Test Set Accuracy:** {final_accuracy:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred_tuned))

    # Confusion matrix
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_tuned)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Low Risk (0)', 'High Risk (1)'],
                yticklabels=['Low Risk (0)', 'High Risk (1)'], ax=ax)
    ax.set_title(f'Tuned Model Confusion Matrix (C={grid_search.best_params_["C"]})')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    st.pyplot(fig_cm)



    feature_names = X_train.columns
    coefficients = best_log_reg.coef_[0]
    odds_ratios = np.exp(coefficients)
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients.round(4),
        'Odds Ratio (e^Coeff)': odds_ratios.round(4)
    }).sort_values(by='Odds Ratio (e^Coeff)', ascending=False)

    st.dataframe(results_df)

    fig_or, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Odds Ratio (e^Coeff)', y='Feature', data=results_df, palette='viridis', ax=ax)
    ax.axvline(1, color='red', linestyle='--')
    ax.set_title('Tuned Model Odds Ratios for Predicting High Addiction Risk')
    ax.set_xlabel('Odds Ratio (e^Coefficient)')
    ax.set_ylabel('Feature')
    st.pyplot(fig_or)



# step 5
with st.expander("""## Step 5. Interpretation & discussion of results"""):
    st.markdown("""
    #### Accuracy and cross validation
    
    The model achieved an accuracy of 98.58%, indicating that it correctly classified nearly all students into the appropriate High Risk or Low Risk addiction categories.

    This exceptionally high accuracy closely matches the cross-validation result (99.59%), suggesting the model has strong generalization ability and did not overfit during training. The minimal performance gap between training and testing further validates the regularization parameter (C = 1.0) as a good balance between bias and variance.


    #### Classification report
    The classification report provides a deeper look into the model’s performance for each class:

    Precision (0.98–0.99): When the model predicts a student as “High Risk,” it is correct almost every time.

    Recall (0.97–0.99): The model successfully identifies nearly all true high-risk students, missing very few.

    F1-score (0.97–0.99): The harmonic mean of precision and recall indicates balanced, reliable predictions across both categories.

    The macro and weighted averages are also high (0.98–0.99), confirming that the model performs consistently well even if one class has fewer samples. Overall, these metrics suggest that logistic regression is a strong classifier for detecting social media addiction risk in this dataset.


    #### Confusion matrix
    The confusion matrix confirms high performance, showing very few misclassifications. There are 2 false negatives and only 1 false positive.


    #### Coefficients and Odds Ratio (RELEVANCE TO SOCIAL MEDIA ADDICTION)
    An odds ratio > 1 increases the likelihood of high addiction risk, while an odds ratio < 1 decreases it.

    Key findings from the tuned model:

    | Rank (Category)       | Feature                           | Odds Ratio | Interpretation                                                                                                                                                      |
    |------------------------|-----------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | 1 (Strongest Increase) | **Conflicts_Over_Social_Media**       | 31.10       | A 1-unit increase in conflicts increases the odds of being high risk by over 31×. **This is the most powerful predictor of addiction risk.**                            |
    | 2 (Major Increase)     | Avg_Daily_Usage_Hours             | 2.76        | A 1-unit increase (≈1 standard deviation) in usage hours increases the odds of being high risk by 2.76×.                                                            |
    | 3 (Strong Decrease)    | **Mental_Health_Score**               | 0.07        | A 1-unit increase (≈1 standard deviation) in mental health score reduces the odds of being high risk by 93% (≈1 − 0.07). **This shows mental health is the strongest protective factor.** |
    | 4 (Platform Specific)  | Most_Used_Platform_Grouped_TikTok | 2.46        | Students whose most-used platform is TikTok are nearly 2.5× more likely to be high risk compared to the baseline group (Facebook).                                  |
    | 5 (Academic Level)     | Academic_Level_High School        | 1.48        | High school students are 48% more likely to be high risk than the baseline (Graduate) students.                                                                     |

    #### Odds Ratios Bar Chart

    The bar plot visually represents these influences, clearly showing the large positive influence of conflicts and usage hours, and the strong negative influence of mental health score and sufficient sleep.

    These results provide interpretable, actionable insights: 

    **Students reporting frequent conflicts due to social media, heavy usage, and poor sleep or mental health are at the highest risk for social media addiction, while students with good mental health are at the lowest risk for social media addiction**


    #### Limitations and Model Robustness

    Data Limitations

    * Self-Reporting: almost all of the features are subjective to the subjects own reporting. There is most likely over and under reporting which reduces data reliability.


    Model Limitations

    * Binary Target Simplification: Converting Addicted_Score into a binary variable (High vs. Low risk) simplifies a continuous spectrum of behavior. This may mask subtle differences between moderate and extreme cases of addiction.

    * Linear Assumptions: Logistic regression assumes a linear relationship between predictors and the log-odds of the outcome. Real-world social behavior is often more complex and may include nonlinear effects or interactions between variables (e.g., the combined effect of poor sleep and high screen time).


    Model Robustness

    * Overfitting: Although regularization and cross-validation reduced overfitting, the near-perfect accuracy suggests that some features may still be tightly correlated or overly specific to this dataset.


    Future Improvements

    * Collect More Diverse Data: Expanding the dataset to include multiple schools, age groups, and countries would improve generalizability.

    * Model Nonlinearities: Future work could apply more flexible models like Random Forests, Gradient Boosting Machines, or Support Vector Machines to capture complex relationships.

    * Feature Engineering: Introducing interaction terms (e.g., Sleep Hours × Avg_Daily_Usage) or new behavioral metrics (such as frequency of social app switching) may uncover additional predictive patterns.

    * Longitudinal Analysis: Tracking changes in social media use and mental health over time would allow for a more causal interpretation rather than a single-point correlation.
    """)
