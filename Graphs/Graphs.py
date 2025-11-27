import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler 
from matplotlib.patches import Patch


file_path = r'E:\Documents\9no Semestre\Algoritmos de Inteligencia Artificial\FinalProject\Grpahs\creditcard.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please ensure the dataset is in the correct location.")
    exit()



# 2. Standardization of 'Amount'

# Reshape the 'Amount' column for standardization (StandardScaler expects a 2D array)
df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1)) 


# 3. Descriptive Statistical Analysis of the Amount (Mean and Median focus)
legit = df[df['Class'] == 0]['Amount']
fraud = df[df['Class'] == 1]['Amount']
legit_scaled = df[df['Class'] == 0]['Scaled_Amount']
fraud_scaled = df[df['Class'] == 1]['Scaled_Amount']

mean_legit = legit.mean()
mean_fraud = fraud.mean()
median_legit = legit.median()
median_fraud = fraud.median()

print("\n--- Descriptive Statistics for Transaction Amount ---")
print(f"Mean Amount (Legitimate): ${mean_legit:.2f}")
print(f"Median Amount (Legitimate): ${median_legit:.2f}")
print(f"Mean Amount (Fraudulent): ${mean_fraud:.2f}")
print(f"Median Amount (Fraudulent): ${median_fraud:.2f}")
print("-" * 50)

# 4. Visualization of Amount Distributions

sns.set_style("whitegrid")

# Plot 1: Boxplot - Highlighting the Mean and Median (Zoomed)
plt.figure(figsize=(10, 6))

# Boxplot 1: Shows the main body and standard outliers
sns.boxplot(x='Class', y='Amount', hue='Class', data=df, palette=['skyblue', 'salmon'], whis=1.5, linewidth=0.5, legend=False)
# Boxplot 2 (overlayed): Specifically to show the mean line 
sns.boxplot(x='Class', y='Amount', hue='Class', data=df, palette=['skyblue', 'salmon'], showfliers=False, showmeans=True, meanline=True, legend=False)

plt.xticks([0, 1], ['Legitimate Transactions (0)', 'Fraudulent Transactions (1)'])
plt.title('Distribution of Transaction Amount by Class (Mean and Median Comparison)', fontsize=16)
plt.ylabel('Transaction Amount ($)', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.ylim(0, 300)

# Add the calculated MEANS as horizontal lines (Dashed lines)
plt.axhline(mean_legit, color='blue', linestyle='--', linewidth=1, label=f'Mean Legitimate: ${mean_legit:.2f}')
plt.axhline(mean_fraud, color='red', linestyle='--', linewidth=1, label=f'Mean Fraudulent: ${mean_fraud:.2f}')

# Add the calculated MEDIANS as horizontal lines (Dotted lines)
plt.axhline(median_legit, color='navy', linestyle=':', linewidth=1.5, label=f'Median Legitimate: ${median_legit:.2f}')
plt.axhline(median_fraud, color='darkred', linestyle=':', linewidth=1.5, label=f'Median Fraudulent: ${median_fraud:.2f}')


plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Distribution (KDE Plot) - Showing Mean and Median
plt.figure(figsize=(10, 6))
sns.kdeplot(legit, label='Legitimate (0)', color='skyblue', fill=True, log_scale=False)
sns.kdeplot(fraud, label='Fraudulent (1)', color='salmon', fill=True, log_scale=False)

# Add vertical lines for the means (Dashed lines)
plt.axvline(mean_legit, color='blue', linestyle='--', linewidth=1, label=f'Mean Legit: ${mean_legit:.2f}')
plt.axvline(mean_fraud, color='red', linestyle='--', linewidth=1, label=f'Mean Fraud: ${mean_fraud:.2f}')

# Add vertical lines for the medians (Dotted lines)
plt.axvline(median_legit, color='navy', linestyle=':', linewidth=1.5, label=f'Median Legit: ${median_legit:.2f}')
plt.axvline(median_fraud, color='darkred', linestyle=':', linewidth=1.5, label=f'Median Fraud: ${median_fraud:.2f}')

plt.title('Frequency Distribution of Transaction Amount by Class', fontsize=16)
plt.xlim(0, 1000)
plt.xlabel('Transaction Amount ($)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: Histogram of Amount (Right Skew Visualization)
plt.figure(figsize=(10, 6))
# We use histplot to visualize the distribution of all transaction amounts.
sns.histplot(df['Amount'], bins=100, kde=True, color='#2C7AE1', alpha=0.7)
plt.title('Histogram of Transaction Amount (Pre-Standardization)', fontsize=16)
plt.xlabel('Transaction Amount ($)', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
# Limit the x-axis to focus on the area where most data is concentrated (low amounts)
plt.xlim(0, 2000) 
plt.tight_layout()
plt.show()


# Plot 4: Standardized Boxplot - Identifying Extreme Outliers 
plt.figure(figsize=(10, 6))
# Using Scaled_Amount to show outliers in standard deviation units
sns.boxplot(x='Class', y='Scaled_Amount', hue='Class', data=df, palette=['skyblue', 'salmon'], legend=False)

plt.xticks([0, 1], ['Legitimate Transactions (0)', 'Fraudulent Transactions (1)'])
plt.title('Standardized Boxplot of Transaction Amount by Class (Z-Scores)', fontsize=16)
plt.ylabel('Standardized Transaction Amount (Z-Score)', fontsize=12)
plt.xlabel('Class', fontsize=12)

# Limit the y-axis to clearly see the distribution; high Z-scores indicate extreme outliers
plt.ylim(-5, 50) 
plt.tight_layout()
plt.show()



# 5. PCA Components Mean Comparison
print("\n" + "="*50)
print("5. PCA Components Mean Comparison (All V_i Components)")
print("Visualizing the difference in means for all PCA features (V1 to V28) between classes.")

# Define ALL PCA features (V1 to V28)
pca_cols = [f'V{i}' for i in range(1, 29)]

# Define the key features to highlight
key_pca_cols = ['V1', 'V3', 'V4', 'V10', 'V12', 'V14', 'V17']

# Calculate the mean for these columns, grouped by Class
mean_comparison = df.groupby('Class')[pca_cols].mean().T.reset_index()
mean_comparison.columns = ['Feature', 'Mean_Legitimate', 'Mean_Fraud']

# Prepare data for plotting (melt the DataFrame)
mean_comparison_melted = mean_comparison.melt(id_vars='Feature', 
                                              value_vars=['Mean_Legitimate', 'Mean_Fraud'],
                                              var_name='Class_Type', 
                                              value_name='Mean_Value')

# Map Class_Type for plot labels
class_map = {'Mean_Legitimate': 'Legitimate', 'Mean_Fraud': 'Fraudulent'}
mean_comparison_melted['Class_Type'] = mean_comparison_melted['Class_Type'].map(class_map)


# Define standard and highlight colors
LEGIT_COLOR = 'skyblue'
FRAUD_COLOR = 'salmon'
KEY_LEGIT_COLOR = 'mediumblue' 
KEY_FRAUD_COLOR = 'darkred'   

# Create the plot - Increased figure size to accommodate all 28 bars
plt.figure(figsize=(18, 7))
ax = sns.barplot(x='Feature', y='Mean_Value', hue='Class_Type', data=mean_comparison_melted, palette=[LEGIT_COLOR, FRAUD_COLOR])

# Add horizontal line at zero for reference (PCA components are centered)
plt.axhline(0, color='gray', linestyle='-', linewidth=0.8)

# Get the unique feature names once
feature_names = mean_comparison_melted['Feature'].unique()
num_features = len(feature_names)

# Logic to change the color of the key PCA components
for i, bar in enumerate(ax.patches):
    # Determine the feature index (0 to 27). Since there are 2 bars per feature, we divide by 2.
    feature_index = i // 2
    
    # We only need to process up to the last feature's second bar.
    if feature_index >= num_features:
        break
    
    # Determine the class type (0: Legitimate, 1: Fraudulent)
    class_type_index = i % 2 
    
    # Get the feature name using the valid index
    feature_name = feature_names[feature_index]
    
    if feature_name in key_pca_cols:
        # Check if it's the 'Legitimate' bar (index 0) or 'Fraudulent' bar (index 1)
        if class_type_index == 0:
            bar.set_color(KEY_LEGIT_COLOR)
        else: # class_type_index == 1
            bar.set_color(KEY_FRAUD_COLOR)

# Create custom legend entries to show the highlight colors
legend_handles = [
    Patch(facecolor=LEGIT_COLOR, label='Legitimate (Other V_i)'),
    Patch(facecolor=FRAUD_COLOR, label='Fraudulent (Other V_i)'),
    Patch(facecolor=KEY_LEGIT_COLOR, label='Legitimate (Key V_i)'),
    Patch(facecolor=KEY_FRAUD_COLOR, label='Fraudulent (Key V_i)')
]

# Set English labels and title
plt.title('Mean Difference Across All PCA Components (V1 to V28)', fontsize=16)
plt.ylabel('Average Value of PCA Component', fontsize=12)
plt.xlabel('PCA Component', fontsize=12)

# Use the custom legend
plt.legend(handles=legend_handles, title='Transaction Class and Relevance')
plt.xticks(rotation=45, ha='right') # Rotate X-labels for better readability
plt.tight_layout()
plt.show()


# 6. Overlapping KDE Plots for Highly Correlated Variables
print("\n" + "="*50)
print("6. Overlapping Kernel Density (KDE) Plots for Highly Correlated Variables")
print("Visualizing class separation for the most discriminatory PCA variables.")

# Recalculate and sort correlations by absolute value with 'Class'
# Exclude 'Time', 'Amount', and the target 'Class' itself
pca_features = [f'V{i}' for i in range(1, 29)]
correlations = df[pca_features + ['Class']].corr()['Class'].abs().sort_values(ascending=False)

# Select the top 7 highly correlated features (excluding 'Class' itself if it somehow showed up)
top_n = 7
top_correlated_vars = correlations.index[1:top_n + 1].tolist()

# Prepare the plot grid (e.g., 4 rows x 2 columns for 7 variables)
fig, axes = plt.subplots(4, 2, figsize=(16, 24))
axes = axes.flatten() # Flatten the 4x2 array for easy indexing

for i, var in enumerate(top_correlated_vars):
    ax = axes[i]
    
    # KDE Plot for Legitimate transactions (Class=0)
    sns.kdeplot(df[df['Class'] == 0][var], 
                label='Legitimate Transactions (0)', 
                color='skyblue', 
                fill=True, 
                alpha=0.6, 
                ax=ax)
    
    # KDE Plot for Fraudulent transactions (Class=1)
    sns.kdeplot(df[df['Class'] == 1][var], 
                label='Fraudulent Transactions (1)', 
                color='salmon', 
                fill=True, 
                alpha=0.6, 
                ax=ax)
    
    ax.set_title(f'Density Distribution for {var}', fontsize=14)
    ax.set_xlabel(var, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10)

# Hide any unused subplots (since we have 7 variables and 8 subplots)
for j in range(len(top_correlated_vars), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Kernel Density (KDE) Plots for Highly Correlated PCA Components with "Class"', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()


# 7. Fraud Rate & Volume Analysis Over Time (RESOLVING THE CONTRADICTION)
print("\n" + "="*50)
print("7. Fraud Rate & Volume Analysis Over Time (RESOLVING THE CONTRADICTION)")
print("Plot 7a (Rate) shows high risk at night. Plot 7b (Volume) shows high count during the day.")

SECONDS_IN_DAY = 24 * 60 * 60
SECONDS_IN_HOUR = 60 * 60
# Convert 'Time' to 'Hour_of_Day' (0-23)
df['Hour_of_Day'] = (df['Time'] % SECONDS_IN_DAY) // SECONDS_IN_HOUR

hour_analysis = df.groupby('Hour_of_Day')['Class'].agg(
    Total_Transactions='count',
    Fraudulent_Count='sum'
).reset_index()

# Calculate the Fraud Rate (Fraudulent Count / Total Transactions)
hour_analysis['Fraud_Rate'] = (hour_analysis['Fraudulent_Count'] / hour_analysis['Total_Transactions']) * 100

# --- PLOT 7a: FRAUD RATE (%) ---
# This plot confirms the observation: high risk at night due to low total volume.
# 
plt.figure(figsize=(12, 6))
ax_rate = sns.barplot(x='Hour_of_Day', y='Fraud_Rate', data=hour_analysis, 
                 color='skyblue', alpha=0.8) # Default color

# Logic: Highlight high risk (Night) vs lower risk (Day) based on RATE
for i, bar in enumerate(ax_rate.patches):
    hour = hour_analysis['Hour_of_Day'].iloc[i]
    # Night/Early Morning (High Fraud Rate: 22h to 6h)
    if hour >= 22 or hour <= 6: 
        bar.set_color('#CC0000') # Dark Red for High Risk Rate
    # Day (Lower Fraud Rate: 7h to 21h)
    else:
        bar.set_color('skyblue') # Blue for Lower Risk Rate

for p in ax_rate.patches:
    ax_rate.annotate(f'{p.get_height():.2f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=8, 
                color='black', 
                xytext=(0, 5), 
                textcoords='offset points', 
                rotation=90)

plt.title('Plot 7a: Fraud Rate (%) per Hour of Day (Relative Risk)', fontsize=16)
plt.xlabel('Hour of Day (0 = 12:00 AM)', fontsize=12)
plt.ylabel('Fraud Rate (%)', fontsize=12)
plt.xticks(np.arange(0, 24, 1))
plt.grid(axis='y', linestyle='--', alpha=0.6)

legend_handles_rate = [
    Patch(facecolor='skyblue', label='Low Risk (Daytime - High legitimate volume)'),
    Patch(facecolor='#CC0000', label='High Risk (Night - Low legitimate volume)')
]
plt.legend(handles=legend_handles_rate, loc='upper right', title='Risk Level')
plt.tight_layout()
plt.show()

# --- PLOT 7b: ABSOLUTE FRAUD VOLUME (Fraud Count) ---
# This plot justifies the statement: "fraudulent activity predominates during the day".
# 
plt.figure(figsize=(12, 6))
ax_count = sns.barplot(x='Hour_of_Day', y='Fraudulent_Count', data=hour_analysis, 
                       color='salmon', alpha=0.8)

# Logic: Highlight high count (Day) vs low count (Night) based on VOLUME
for i, bar in enumerate(ax_count.patches):
    hour = hour_analysis['Hour_of_Day'].iloc[i]
    # Day hours (High Absolute Volume: 7h to 21h)
    if hour >= 7 and hour <= 21: 
        bar.set_color('darkred') # Dark Red for High Absolute Volume
    # Night/Early Morning (Low Absolute Volume: 22h to 6h)
    else:
        bar.set_color('salmon') # Salmon for Lower Absolute Volume

for p in ax_count.patches:
    ax_count.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', 
                      va='center', 
                      fontsize=8, 
                      color='black', 
                      xytext=(0, 5), 
                      textcoords='offset points', 
                      rotation=90)

plt.title('Plot 7b: Absolute Fraud Count per Hour of Day (Activity Volume)', fontsize=16)
plt.xlabel('Hour of Day (0 = 12:00 AM)', fontsize=12)
plt.ylabel('Absolute Fraud Count', fontsize=12)
plt.xticks(np.arange(0, 24, 1))
plt.grid(axis='y', linestyle='--', alpha=0.6)

legend_handles_count = [
    Patch(facecolor='salmon', label='Low Volume (Night)'),
    Patch(facecolor='darkred', label='High Volume (Day - Activity Predominance)')
]
plt.legend(handles=legend_handles_count, loc='upper right', title='Fraud Volume')
plt.tight_layout()
plt.show()


# 8. Scatter Plots for Highly Correlated Features
print("\n" + "="*50)
print("8. Scatter Plots for Highly Correlated Features")
print("Visualizing class separation in 2D space using the most discriminatory V-features.")

# Re-using the most correlated variables calculated in Section 6.
# Selecting four key pairs for a 2x2 grid.
scatter_pairs = [('V17', 'V14'), ('V12', 'V10'), ('V4', 'V3'), ('V1', 'V3')] 

plt.figure(figsize=(16, 14))

for i, (x_var, y_var) in enumerate(scatter_pairs):
    plt.subplot(2, 2, i + 1)
    
    # Using seaborn scatterplot. Setting low transparency for the majority class.
    sns.scatterplot(x=df[x_var], y=df[y_var], 
                    hue=df['Class'], 
                    style=df['Class'],
                    palette=['skyblue', 'red'], 
                    alpha=0.2, # Low transparency for the legitimate class (blue)
                    s=5, # Small marker size
                    legend='full')
    
    # To ensure visibility, we plot the fraudulent class again with higher opacity and size.
    fraud_data = df[df['Class'] == 1]
    plt.scatter(fraud_data[x_var], fraud_data[y_var], color='red', s=20, label='Fraudulent (1)', alpha=0.8)

    plt.title(f'Scatter Plot: {x_var} vs {y_var} (Colored by Class)', fontsize=14)
    plt.xlabel(x_var, fontsize=12)
    plt.ylabel(y_var, fontsize=12)
    
    # Customize the legend to show only the Legitimate and Fraudulent entries
    custom_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=5, label='Legitimate (0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Fraudulent (1)')
    ]
    plt.legend(handles=custom_handles, title='Transaction Class', loc='best')


plt.suptitle('2D Scatter Plots of Key PCA Components for Class Separation', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()


# ====================================================================================================================

# 9. Amount Segmentation by Quartiles
print("\n" + "="*50)
print("9. Amount Segmentation by Quartiles")
print("Analyzing the percentage of fraud within transaction amount segments.")

# 9.1 Calculate Quartiles of 'Amount'
# We use the 0th and 100th percentiles to ensure we cover the entire range
bins = [df['Amount'].min(), 
        df['Amount'].quantile(0.25), 
        df['Amount'].quantile(0.50), 
        df['Amount'].quantile(0.75), 
        df['Amount'].max()]

# Label the quartiles
labels = ['Q1 (Low)', 'Q2 (Med-Low)', 'Q3 (Med-High)', 'Q4 (High)']

# 9.2 Segment the dataset
# We use 'include_lowest=True' to include the minimum value
df['Amount_Quartile'] = pd.cut(df['Amount'], bins=bins, labels=labels, include_lowest=True, duplicates='drop')

# 9.3 Calculate statistics per segment
quartile_analysis = df.groupby('Amount_Quartile')['Class'].agg(
    Total_Transactions='count',
    Fraudulent_Count='sum'
).reset_index()

# Calculate the fraud percentage
quartile_analysis['Fraud_Percentage'] = (quartile_analysis['Fraudulent_Count'] / quartile_analysis['Total_Transactions']) * 100

# Add the amount range information for printing
quartile_analysis['Amount_Range'] = [
    f"${bins[0]:.2f} - ${bins[1]:.2f}",
    f"${bins[1]:.2f} - ${bins[2]:.2f}",
    f"${bins[2]:.2f} - ${bins[3]:.2f}",
    f"${bins[3]:.2f} - ${bins[4]:.2f}"
]

print("\n--- Fraud Analysis by Amount Quartile ---")
print(quartile_analysis[['Amount_Quartile', 'Amount_Range', 'Total_Transactions', 'Fraudulent_Count', 'Fraud_Percentage']].to_string(index=False))
print("-" * 50)


# 9.4 Visualization
plt.figure(figsize=(10, 6))

# Create the bar chart with the fraud percentage
ax = sns.barplot(x='Amount_Quartile', y='Fraud_Percentage', data=quartile_analysis, palette='viridis')

# Add text labels on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                fontsize=10, 
                color='black',
                xytext=(0, 5), 
                textcoords='offset points')

plt.title('Fraud Percentage by Transaction Amount Quartile', fontsize=16)
plt.xlabel('Amount Quartile Segment', fontsize=12)
plt.ylabel('Fraud Percentage (%)', fontsize=12)
plt.tight_layout()
plt.show()



# 10. Correlation Heatmap of All Features (Time, V1-V28, Amount, Class)
print("\n" + "="*50)
print("10. Correlation Heatmap of All Features")
print("Visualizing the correlation structure among all features, including Time, Amount, and Class.")

# Select all columns
all_cols_corr = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

# Calculate the correlation matrix
corr_matrix = df[all_cols_corr].corr()

plt.figure(figsize=(24, 22))
sns.heatmap(corr_matrix, 
            cmap='coolwarm', 
            annot=True, # Show correlation values for better insight
            fmt=".2f",
            linewidths=.5, 
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Correlation Heatmap Among All Features', fontsize=20)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
