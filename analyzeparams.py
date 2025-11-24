import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')


def analyze_hyperparameter_effects(excel_file_path, sheet_name='val'):
    """
    Analyze the effect of hyperparameters on recall@10

    Parameters:
    excel_file_path: str - path to Excel file
    sheet_name: str - sheet name (default: 'val')
    """

    # Load data from val sheet
    print("Loading data from 'val' sheet...")
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Define hyperparameter columns and target column based on your data
    hyperparam_cols = ['overlap', 'lr', 'embed_dim', 'num_groups', 'num_prev_items',
                       'prediction_loss_weight', 'window_size', 'smoothing']
    recall_col = 'Recall@10'

    # Check if all columns exist
    missing_cols = [col for col in hyperparam_cols + [recall_col] if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    print(f"\nAnalyzing effect of {len(hyperparam_cols)} hyperparameters on {recall_col}")
    print(f"Hyperparameters: {hyperparam_cols}")

    print(f"\nAnalyzing effect of hyperparameters on {recall_col}")
    print(f"Hyperparameters: {hyperparam_cols}")

    # Basic statistics
    print(f"\n{recall_col} statistics:")
    print(df[recall_col].describe())

    # 1. Correlation Analysis
    print("\n" + "=" * 50)
    print("1. CORRELATION ANALYSIS")
    print("=" * 50)

    correlations = {}
    for param in hyperparam_cols:
        corr, p_value = pearsonr(df[param], df[recall_col])
        correlations[param] = {'correlation': corr, 'p_value': p_value}
        print(f"{param}: correlation = {corr:.4f}, p-value = {p_value:.4f}")

    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)

    print("\nRanked by correlation strength:")
    for param, stats_dict in sorted_correlations:
        corr = stats_dict['correlation']
        p_val = stats_dict['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{param}: {corr:.4f} {significance}")

    # 2. ANOVA Analysis (for categorical hyperparameters)
    print("\n" + "=" * 50)
    print("2. ANOVA ANALYSIS")
    print("=" * 50)

    anova_results = {}
    for param in hyperparam_cols:
        # Group by parameter values
        unique_values = df[param].nunique()
        if unique_values <= 10:  # Only for parameters with limited unique values
            groups = [df[df[param] == val][recall_col].values for val in df[param].unique()]
            groups = [group for group in groups if len(group) > 0]  # Remove empty groups

            if len(groups) > 1:
                f_stat, p_value = f_oneway(*groups)
                anova_results[param] = {'f_statistic': f_stat, 'p_value': p_value}
                print(f"{param}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

    # 3. Feature Importance using Random Forest
    print("\n" + "=" * 50)
    print("3. FEATURE IMPORTANCE (Random Forest)")
    print("=" * 50)

    X = df[hyperparam_cols]
    y = df[recall_col]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importance = dict(zip(hyperparam_cols, rf.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print("Feature Importance Rankings:")
    for param, importance in sorted_importance:
        print(f"{param}: {importance:.4f}")

    # 4. Min/Max Recall@10 for Each Parameter Value
    print("\n" + "=" * 50)
    print("4. MIN/MAX RECALL@10 FOR EACH PARAMETER VALUE")
    print("=" * 50)

    param_value_analysis = analyze_parameter_values(df, hyperparam_cols, recall_col)

    # 5. Sensitivity Analysis
    print("\n" + "=" * 50)
    print("5. SENSITIVITY ANALYSIS")
    print("=" * 50)

    sensitivity_results = {}
    for param in hyperparam_cols:
        param_range = df[param].max() - df[param].min()
        recall_range_for_param = df.groupby(param)[recall_col].agg(['min', 'max'])
        max_recall_diff = (recall_range_for_param['max'] - recall_range_for_param['min']).max()

        # Calculate sensitivity as max recall difference / parameter range
        if param_range > 0:
            sensitivity = max_recall_diff / param_range
            sensitivity_results[param] = {
                'max_recall_difference': max_recall_diff,
                'param_range': param_range,
                'sensitivity': sensitivity
            }
            print(f"{param}: Max recall difference = {max_recall_diff:.4f}, Sensitivity = {sensitivity:.6f}")

    # 6. Summary and Recommendations
    print("\n" + "=" * 50)
    print("6. SUMMARY AND RECOMMENDATIONS")
    print("=" * 50)

    # Create ranking scores
    param_scores = {}
    for param in hyperparam_cols:
        score = 0

        # Correlation score (0-3 points)
        abs_corr = abs(correlations[param]['correlation'])
        if abs_corr > 0.5:
            score += 3
        elif abs_corr > 0.3:
            score += 2
        elif abs_corr > 0.1:
            score += 1

        # P-value score (0-2 points)
        if correlations[param]['p_value'] < 0.01:
            score += 2
        elif correlations[param]['p_value'] < 0.05:
            score += 1

        # Feature importance score (0-2 points)
        importance_rank = [p for p, _ in sorted_importance].index(param)
        if importance_rank < 2:
            score += 2
        elif importance_rank < 4:
            score += 1

        param_scores[param] = score

    # Sort parameters by total score
    sorted_params = sorted(param_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nParameter Significance Ranking (Higher score = More significant):")
    for i, (param, score) in enumerate(sorted_params, 1):
        print(f"{i}. {param}: Score = {score}/7")

    # Recommendations
    print("\nRECOMMENDations:")
    high_impact = [param for param, score in sorted_params if score >= 5]
    medium_impact = [param for param, score in sorted_params if 3 <= score < 5]
    low_impact = [param for param, score in sorted_params if score < 3]

    if high_impact:
        print(f"HIGH IMPACT (Keep for tuning): {high_impact}")
    if medium_impact:
        print(f"MEDIUM IMPACT (Consider keeping): {medium_impact}")
    if low_impact:
        print(f"LOW IMPACT (Consider filtering out): {low_impact}")

    # 7. Visualization
    create_visualizations(df, hyperparam_cols, recall_col, correlations, feature_importance)

    return {
        'correlations': correlations,
        'anova_results': anova_results,
        'feature_importance': feature_importance,
        'sensitivity_results': sensitivity_results,
        'parameter_scores': param_scores,
        'param_value_analysis': param_value_analysis
    }


def analyze_parameter_values(df, hyperparam_cols, recall_col):
    """
    Analyze min/max recall@10 for each value of each parameter
    """
    param_value_results = {}

    for param in hyperparam_cols:
        print(f"\n{param.upper()}:")
        print("-" * (len(param) + 1))

        # Group by parameter values and calculate stats
        grouped = df.groupby(param)[recall_col].agg(['count', 'min', 'max', 'mean', 'std']).round(4)
        grouped['range'] = (grouped['max'] - grouped['min']).round(4)

        # Sort by parameter value for better readability
        grouped = grouped.sort_index()

        # Store results
        param_value_results[param] = grouped

        # Display results
        print(f"{'Value':<12} {'Count':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Range':<8}")
        print("-" * 65)

        for value, row in grouped.iterrows():
            print(f"{str(value):<12} {int(row['count']):<8} {row['min']:<8.4f} {row['max']:<8.4f} "
                  f"{row['mean']:<8.4f} {row['std']:<8.4f} {row['range']:<8.4f}")

        # Identify best and worst performing values
        best_value = grouped['max'].idxmax()
        worst_value = grouped['min'].idxmin()
        best_recall = grouped.loc[best_value, 'max']
        worst_recall = grouped.loc[worst_value, 'min']

        print(f"\nBest performing value: {param}={best_value} (max recall = {best_recall:.4f})")
        print(f"Worst performing value: {param}={worst_value} (min recall = {worst_recall:.4f})")

        # Calculate impact - difference between best max and worst min
        total_impact = best_recall - worst_recall
        print(f"Total impact range: {total_impact:.4f}")

        # Show values with highest variance (most inconsistent)
        if len(grouped) > 1:
            highest_variance_value = grouped['std'].idxmax()
            print(
                f"Most variable value: {param}={highest_variance_value} (std = {grouped.loc[highest_variance_value, 'std']:.4f})")

    return param_value_results


def create_visualizations(df, hyperparam_cols, recall_col, correlations, feature_importance):
    """Create visualizations for hyperparameter analysis"""

    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    # 1. Correlation heatmap
    plt.subplot(2, 3, 1)
    corr_matrix = df[hyperparam_cols + [recall_col]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 2. Correlation bar plot
    plt.subplot(2, 3, 2)
    corr_values = [correlations[param]['correlation'] for param in hyperparam_cols]
    colors = ['red' if x < 0 else 'blue' for x in corr_values]
    plt.bar(range(len(hyperparam_cols)), corr_values, color=colors, alpha=0.7)
    plt.xlabel('Hyperparameters')
    plt.ylabel('Correlation with Recall@10')
    plt.title('Correlation Strength')
    plt.xticks(range(len(hyperparam_cols)), hyperparam_cols, rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 3. Feature importance
    plt.subplot(2, 3, 3)
    importance_values = [feature_importance[param] for param in hyperparam_cols]
    plt.bar(range(len(hyperparam_cols)), importance_values, color='green', alpha=0.7)
    plt.xlabel('Hyperparameters')
    plt.ylabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.xticks(range(len(hyperparam_cols)), hyperparam_cols, rotation=45)

    # 4. Scatter plots for top 3 most correlated parameters
    top_3_params = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:3]

    for i, (param, _) in enumerate(top_3_params):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(df[param], df[recall_col], alpha=0.6)
        plt.xlabel(param)
        plt.ylabel(recall_col)
        plt.title(f'{param} vs {recall_col}')

        # Add trend line
        z = np.polyfit(df[param], df[recall_col], 1)
        p = np.poly1d(z)
        plt.plot(df[param], p(df[param]), "r--", alpha=0.8)

    plt.tight_layout()
    plt.show()


# Example usage for your specific data:
if __name__ == "__main__":
    # Replace 'your_file.xlsx' with your actual file path
    file_path = 'tgra_hypertune_resultsdell.xlsx'

    try:
        # This will automatically use the 'val' sheet and your specific columns
        results = analyze_hyperparameter_effects(file_path)
        print("\nAnalysis completed successfully!")

        # You can also analyze other metrics if needed:
        # analyze_other_metrics(file_path)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please update the file_path variable with the correct path to your Excel file")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your Excel file format and column names")


# def analyze_other_metrics(file_path):
#     """
#     Optional: Analyze effects on other metrics too (MRR@10, NDCG@10, etc.)
#     """
#     other_metrics = ['MRR@10', 'NDCG@10', 'Recall@20', 'MRR@20', 'NDCG@20']
#
#     for metric in other_metrics:
#         print(f"\n{'=' * 60}")
#         print(f"ANALYSIS FOR {metric}")
#         print('=' * 60)
#
#         df = pd.read_excel(file_path, sheet_name='val')
#         hyperparam_cols = ['overlap', 'lr', 'embed_dim', 'num_groups', 'num_prev_items',
#                            'prediction_loss_weight', 'window_size', 'smoothing']
#
#         # Quick correlation analysis
#         correlations = {}
#         for param in hyperparam_cols:
#             if param in df.columns and metric in df.columns:
#                 corr, p_value = pearsonr(df[param], df[metric])
#                 correlations[param] = {'correlation': corr, 'p_value': p_value}
#
#         # Sort and display
#         sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
#
#         print(f"Top correlations with {metric}:")
#         for param, stats_dict in sorted_correlations[:5]:
#             corr = stats_dict['correlation']
#             p_val = stats_dict['p_value']
#             significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#             print(f"  {param}: {corr:.4f} {significance}")
#
#
# # Additional utility function to analyze specific parameter values
# def detailed_parameter_analysis(file_path, parameter_name, sheet_name='val'):
#     """
#     Get detailed analysis for a specific parameter
#
#     Parameters:
#     file_path: str - path to Excel file
#     parameter_name: str - name of the parameter to analyze in detail
#     sheet_name: str - sheet name (default: 'val')
#     """
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     recall_col = 'Recall@10'
#
#     if parameter_name not in df.columns:
#         print(f"Parameter '{parameter_name}' not found in data")
#         return
#
#     print(f"DETAILED ANALYSIS FOR {parameter_name.upper()}")
#     print("=" * 50)
#
#     # Get all experiments for each parameter value
#     grouped = df.groupby(parameter_name)
#
#     for value, group in grouped:
#         print(f"\n{parameter_name} = {value} ({len(group)} experiments):")
#         recall_values = group[recall_col].values
#
#         print(f"  Recall@10 values: {[f'{x:.4f}' for x in recall_values]}")
#         print(f"  Min: {recall_values.min():.4f}")
#         print(f"  Max: {recall_values.max():.4f}")
#         print(f"  Mean: {recall_values.mean():.4f}")
#         print(f"  Std: {recall_values.std():.4f}")
#
#         # Show the configuration that achieved best recall for this parameter value
#         best_idx = group[recall_col].idxmax()
#         best_config = group.loc[best_idx]
#         print(
#             f"  Best config: {dict(best_config[['overlap', 'lr', 'embed_dim', 'num_groups', 'num_prev_items', 'prediction_loss_weight', 'window_size', 'smoothing']])}")
#         print(f"  Best recall: {best_config[recall_col]:.4f}")
#
#
# # Simple function to just get min/max table for all parameters
# def get_parameter_minmax_table(file_path, sheet_name='val'):
#     """
#     Get a simple min/max table for all parameters
#     """
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     hyperparam_cols = ['overlap', 'lr', 'embed_dim', 'num_groups', 'num_prev_items',
#                        'prediction_loss_weight', 'window_size', 'smoothing']
#     recall_col = 'Recall@10'
#
#     print("PARAMETER VALUE PERFORMANCE SUMMARY")
#     print("=" * 50)
#
#     summary_data = []
#
#     for param in hyperparam_cols:
#         grouped = df.groupby(param)[recall_col].agg(['min', 'max', 'mean', 'count'])
#
#         # Find the best and worst values
#         best_value = grouped['max'].idxmax()
#         worst_value = grouped['min'].idxmin()
#
#         summary_data.append({
#             'Parameter': param,
#             'Best_Value': best_value,
#             'Best_Max_Recall': grouped.loc[best_value, 'max'],
#             'Worst_Value': worst_value,
#             'Worst_Min_Recall': grouped.loc[worst_value, 'min'],
#             'Impact_Range': grouped.loc[best_value, 'max'] - grouped.loc[worst_value, 'min']
#         })
#
#     # Create summary DataFrame
#     summary_df = pd.DataFrame(summary_data)
#     summary_df = summary_df.sort_values('Impact_Range', ascending=False)
#
#     print(
#         f"{'Parameter':<20} {'Best Value':<12} {'Best Recall':<12} {'Worst Value':<12} {'Worst Recall':<12} {'Impact':<8}")
#     print("-" * 85)
#
#     for _, row in summary_df.iterrows():
#         print(f"{row['Parameter']:<20} {str(row['Best_Value']):<12} {row['Best_Max_Recall']:<12.4f} "
#               f"{str(row['Worst_Value']):<12} {row['Worst_Min_Recall']:<12.4f} {row['Impact_Range']:<8.4f}")
#
#     return summary_df
