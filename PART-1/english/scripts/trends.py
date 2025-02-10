import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# Set up the visual style with larger figure size for better spacing
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [14, 8]  # Increased figure size
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelpad'] = 15  # Increased label padding

# Create vibrant color palette with better contrast
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5']
sns.set_palette(sns.color_palette(colors))

# Create output folder for plots
output_folder = "./evaluation_results/plots"
os.makedirs(output_folder, exist_ok=True)

# Load and prepare the data
results_file = "./evaluation_results/evaluation_summary.csv"
df = pd.read_csv(results_file)

def format_metric_name(metric):
    return metric.replace('-', ' ').title()

def plot_with_trend_and_highlights(data, x, y, hue, ax, **kwargs):
    # Plot main trend lines with adjusted alpha for better visibility
    sns.lineplot(data=data, x=x, y=y, hue=hue, style=hue, 
                linewidth=2.5, alpha=0.8, ax=ax, markers=True, 
                marker='o', markersize=8)
    
    # Add shaded confidence region with reduced alpha
    for hue_val in data[hue].unique():
        hue_data = data[data[hue] == hue_val]
        ax.fill_between(hue_data[x], 
                       hue_data[y] - hue_data[y].std(),
                       hue_data[y] + hue_data[y].std(),
                       alpha=0.1)  # Reduced alpha for better clarity
    
    # Highlight best performing points with larger markers
    top_3 = data.nlargest(3, y)
    scatter = ax.scatter(top_3[x], top_3[y], c='gold', s=200, 
                        zorder=5, label='Top Performers', marker='*',
                        edgecolor='black', linewidth=1)
    
    return ax

def add_analysis_annotations(ax, data, x, y, title):
    # Add trend arrow in top right corner instead of top left
    y_vals = data[y].values
    x_vals = data[x].values
    trend = np.polyfit(x_vals, y_vals, 1)[0]
    
    arrow_text = "↗ Increasing" if trend > 0 else "↘ Decreasing"
    ax.text(0.98, 0.98, f"Trend: {arrow_text}", 
            transform=ax.transAxes, 
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
            verticalalignment='top', horizontalalignment='right')
    
    # Add key statistics with adjusted position
    stats_text = (f"Max: {data[y].max():.3f}\n"
                 f"Min: {data[y].min():.3f}\n"
                 f"Range: {(data[y].max() - data[y].min()):.3f}")
    
    ax.text(0.98, 0.80, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
            horizontalalignment='right')

# List of metrics to plot
metrics = ["SimLex-Correlation", "WordSim-Correlation", "Clustering-Silhouette"]

# Create plots for each normalization method
for norm in df["Normalization"].unique():
    df_norm = df[df["Normalization"] == norm]
    
    for metric in metrics:
        # Create figure with extra space for legend
        fig = plt.figure(figsize=(14, 8), facecolor='white')
        # Add subplot with specific geometry to leave space for legend
        ax = fig.add_subplot(111)
        ax.set_facecolor('#f8f9fa')
        
        # Plot main visualization
        plot_with_trend_and_highlights(df_norm, "Dimensions", metric, "Window", ax)
        
        # Customize the plot with adjusted title padding
        ax.set_title(f"{format_metric_name(metric)} vs. Dimensions\n{norm} Normalization", 
                    pad=20, fontsize=14, fontweight='bold', y=1.05)
        ax.set_xlabel("Dimensions", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel(format_metric_name(metric), fontsize=12, fontweight='bold', labelpad=10)
        
        # Add grid with custom styling
        ax.grid(True, linestyle='--', alpha=0.2, color='gray')
        
        # Force x-axis to use integers with proper spacing
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        
        # Add analysis annotations
        add_analysis_annotations(ax, df_norm, "Dimensions", metric, 
                               f"{metric} Analysis for {norm}")
        
        # Add performance zones with better visibility
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.axhspan(y_max - y_range * 0.2, y_max, 
                  color='#90EE90'
                  , alpha=0.5, label='Optimal Zone')
        ax.axhspan(y_min, y_min + y_range * 0.2, 
                  color='#FFB6C1'
                  , alpha=0.5, label='Suboptimal Zone')
        
        # Adjust legend position and style
        legend = ax.legend(title="Window Size", title_fontsize=12, fontsize=10,
                         bbox_to_anchor=(1.15, 1), loc='upper left',
                         framealpha=0.9, edgecolor='gray')
        legend.get_frame().set_boxstyle('round,pad=0.5')
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(right=0.85, top=0.9)
        
        # Save the figure with extra padding
        plot_filename = f"{metric.replace('-', '_')}_vs_Dimensions_{norm}.png"
        plt.savefig(os.path.join(output_folder, plot_filename), 
                   bbox_inches='tight', facecolor='white', 
                   pad_inches=0.3)  # Added padding
        plt.close()
        print(f"Saved {plot_filename}")

# Create comparison plots for fixed dimension
fixed_dimension = 100
df_fixed_dim = df[df["Dimensions"] == fixed_dimension]

for metric in metrics:
    # Create figure with extra space for legend and summary
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#f8f9fa')
    
    # Create enhanced window size plot
    plot_with_trend_and_highlights(df_fixed_dim, "Window", metric, "Normalization", ax)
    
    # Customize the plot with better spacing
    ax.set_title(f"{format_metric_name(metric)} vs. Window Size\nDimensions = {fixed_dimension}", 
                pad=20, fontsize=14, fontweight='bold', y=1.05)
    ax.set_xlabel("Window Size", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel(format_metric_name(metric), fontsize=12, fontweight='bold', labelpad=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    
    # Force x-axis to use integers with proper spacing
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    # Add analysis annotations
    add_analysis_annotations(ax, df_fixed_dim, "Window", metric,
                           f"{metric} Analysis for Window Size")
    
    # Add performance zones with better visibility
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.axhspan(y_max - y_range * 0.2, y_max, 
              color='#90EE90', alpha=0.5, label='Optimal Zone')
    ax.axhspan(y_min, y_min + y_range * 0.2, 
              color='#FFB6C1', alpha=0.5, label='Suboptimal Zone')
    
    # Add summary box with better positioning
    summary_stats = []
    for norm in df_fixed_dim["Normalization"].unique():
        norm_data = df_fixed_dim[df_fixed_dim["Normalization"] == norm][metric]
        summary_stats.append(f"{norm}:\nMean: {norm_data.mean():.3f}")
    
    summary_text = "Performance Summary:\n" + "\n\n".join(summary_stats)
    plt.figtext(1.02, 0.5, summary_text, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9,
                         boxstyle='round,pad=0.5'))
    
    # Adjust legend position and style
    legend = ax.legend(title="Normalization Method", title_fontsize=12, fontsize=10,
                     bbox_to_anchor=(1.15, 1), loc='upper left',
                     framealpha=0.9, edgecolor='gray')
    legend.get_frame().set_boxstyle('round,pad=0.5')
    
    # Adjust layout to prevent cutoff
    plt.subplots_adjust(right=0.85, top=0.9)
    
    # Save the figure with extra padding
    plot_filename = f"{metric.replace('-', '_')}_vs_Window_Dim{fixed_dimension}.png"
    plt.savefig(os.path.join(output_folder, plot_filename), 
                bbox_inches='tight', facecolor='white',
                pad_inches=0.3)  # Added padding
    plt.close()
    print(f"Saved {plot_filename}")

print("All enhanced plots have been saved in the 'plots' folder.")