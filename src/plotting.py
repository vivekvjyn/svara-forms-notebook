from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import numpy as np
import pandas as pd

import scienceplots

plt.style.use(['science'])

titlesize = 6
ticksize = 5
labelsize = 5
gridweight = 0.3
annot_lineweight = 0.5
figsize=(3,1.4)
color='darkorange'
contextcolor='lightslategrey'
rec_color='gainsboro' 


def format_func(value, tick_number):
    minutes = int(value // 60)
    seconds = int(value % 60)
    milliseconds = int((value % 1) * 1000)
    return f'{minutes}:{seconds:02d}.{milliseconds:03d}'


def plot_with_context(annotations, curr_ix, beat_length, timestep, pitch, time, plot_kwargs, plot_path):
    
    curr_row = annotations.loc[curr_ix]
    
    if curr_ix != 0:
        prev_row = annotations.loc[curr_ix-1]
    else:
        prev_row = None
    
    if curr_ix != len(annotations)-1:
        next_row = annotations.loc[curr_ix+1]
    else:
        next_row = None
    
    curri = curr_row.name

    if not prev_row is None:
        prev_text_x = prev_row['start_sec']+(prev_row['end_sec']-prev_row['start_sec'])/2
    
    curr_text_x = curr_row['start_sec']+(curr_row['end_sec']-curr_row['start_sec'])/2
    
    if not next_row is None:
        next_text_x = next_row['start_sec']+(next_row['end_sec']-next_row['start_sec'])/2

    L = curr_row['label']
    LS = format_func(curr_row['start_sec'],1)

    title = f'{L} at {LS}'



    # prev row
    if not prev_row is None:
        start = prev_row['start_sec']
        end = prev_row['end_sec']

        s1_beats = start/beat_length
        s2_beats = end/beat_length

        s1 = round(s1_beats/timestep)
        s2 = round(s2_beats/timestep)

        prev_psamp = pitch[s1:s2]
        prev_tsamp_beats = time[s1:s2]
        prev_tsamp = prev_tsamp_beats*beat_length

        xlim_0 = prev_row['start_sec']
    else:
        prev_psamp = np.array([])
        prev_tsamp = np.array([])
        xlim_0 = curr_row['start_sec']

    # curr row
    start = curr_row['start_sec']
    end = curr_row['end_sec']

    s1_beats = start/beat_length
    s2_beats = end/beat_length

    s1 = round(s1_beats/timestep)
    s2 = round(s2_beats/timestep)

    curr_psamp = pitch[s1:s2]
    curr_tsamp_beats = time[s1:s2]
    curr_tsamp = curr_tsamp_beats*beat_length


    if not next_row is None:
        # next row
        start = next_row['start_sec']
        end = next_row['end_sec']

        s1_beats = start/beat_length
        s2_beats = end/beat_length

        s1 = round(s1_beats/timestep)
        s2 = round(s2_beats/timestep)

        next_psamp = pitch[s1:s2]
        next_tsamp_beats = time[s1:s2]
        next_tsamp = next_tsamp_beats*beat_length

        xlim_1 = next_row['end_sec']
    else:
        next_psamp = np.array([])
        next_tsamp = np.array([])
        xlim_1 = curr_row['end_sec']


    all_samp = np.concatenate([curr_psamp, prev_psamp, next_psamp])

    minp = np.nanmin(all_samp[all_samp != None])-200
    maxp = np.nanmax(all_samp[all_samp != None])+200

    yticks_dict = {k:v for k,v in plot_kwargs['yticks_dict'].items() if v >= minp and v <= maxp}

    # Plot
    plt.close('all')
    fig, ax = plt.subplots(1)
    fig.set_size_inches(figsize[0], figsize[1])
    
    rect = Rectangle((curr_row['start_sec'], minp), curr_row['end_sec']-curr_row['start_sec'], maxp-minp, facecolor=rec_color)
    ax.add_patch(rect)
    

    ax.plot(prev_tsamp, prev_psamp, color=contextcolor)
    ax.plot(curr_tsamp, curr_psamp, color=color)
    ax.plot(next_tsamp, next_psamp, color=contextcolor)
    ax.set_title(title, fontsize=titlesize)

    ax.set_ylabel(f'Cents as svara \npositions', fontsize=labelsize)
    tick_names = list(yticks_dict.keys())
    tick_loc = list(yticks_dict.values())
    ax.set_yticks(tick_loc)
    ax.set_yticklabels(tick_names)
    ax.grid(zorder=8, linestyle='--', linewidth=gridweight)
    
    if not np.isnan(maxp) and not np.isnan(minp):
        ax.set_ylim((minp, maxp))
    
    ax.set_xlim((xlim_0, xlim_1))

    ax.set_title(title, fontsize=titlesize)

    ax.set_xlabel('Time (s)', fontsize=labelsize)

    ## Add text
    text_y = minp + (maxp - minp)*0.8

    if not prev_row is None:
        plt.text(prev_text_x, text_y, prev_row['label'])

    plt.text(curr_text_x, text_y, curr_row['label'])

    if not next_row is None:
        plt.text(next_text_x, text_y, next_row['label'])
    
    plt.axvline(curr_tsamp[0],linestyle='--', lw=0.5, color='black')
    plt.axvline(curr_tsamp[-1],linestyle='--', lw=0.5, color='black')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    plt.xticks(fontsize=ticksize, zorder=2)
    plt.yticks(fontsize=ticksize, zorder=2)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close('all')


def save_boxplot(data, names, subgroups, filename, figsize=(15,4.5), title=None, ylabel=None, xlabel=None, xlim=None):
    """
    Creates and saves a color-coded boxplot for the given list of lists, where each group can belong to a subgroup.

    Parameters:
    - data: List of lists containing numerical data for each category.
    - names: List of strings corresponding to the names of each list in data.
    - subgroups: List of subgroup labels corresponding to each group in data.
    - filename: The name of the file where the plot will be saved.
    """
    # Ensure data is numerical
    data = [np.array(group, dtype=float) for group in data]
    
    # Ensure subgroups is a numpy array (categorical or strings are fine)
    subgroups = np.array(subgroups)
    
    # Ensure names are a list of strings
    names = list(map(str, names))

    # Define unique subgroups and assign colors to them
    unique_subgroups = list(set(subgroups))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_subgroups)))  # Color palette
    
    # Create a color map for subgroups
    color_map = {subgroup: color for subgroup, color in zip(unique_subgroups, colors)}
    
    # Create the boxplot
    plt.figure(figsize=figsize)  # Adjust the size of the figure if needed
    boxprops = dict(linewidth=2)
    
    boxplot = plt.boxplot(data, patch_artist=True, notch=True, boxprops=boxprops)
    
    # Color each box according to its subgroup
    for patch, subgroup in zip(boxplot['boxes'], subgroups):
        patch.set_facecolor(color_map[subgroup])
    
    # Scatter individual data points
    for i, group in enumerate(data, start=1):
        # Add random jitter to x-axis positions to avoid overlap
        x_positions = np.random.normal(i, 0.04, size=len(group))
        plt.scatter(x_positions, group, color=color_map[subgroups[i-1]], alpha=0.6, edgecolor='k', label=subgroups[i-1], zorder=10)
        #plt.scatter(x_positions, group, color='darkred', alpha=0.6, edgecolor='k', label=subgroups[i-1], zorder=10, marker='x')

    # Add labels
    plt.xticks(ticks=range(1, len(names) + 1), labels=names)
    
    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)
    
    if xlim:
        plt.xlim(xlim)

    if ylabel:
        plt.ylabel(ylabel)
    
    # Create the legend
    handles = [plt.Line2D([0], [0], color=color_map[subgroup], lw=4, label=subgroup) 
               for subgroup in unique_subgroups]
    plt.legend(handles=handles)
    plt.grid()
    # Save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory


def save_histogram_with_kde(values, bins=10, title="Histogram with KDE", figsize=(8, 6), xlabel="X-axis", ylabel="Density", filename="histogram_kde.png"):
    """
    Creates a histogram with a KDE curve and saves it to a file.
    
    Args:
        values (list): List of values to plot.
        bins (int): Number of bins for the histogram.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure (width, height).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str): File name to save the plot (default: 'histogram_kde.png').
    """
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot histogram and KDE using seaborn
    sns.histplot(values, bins=bins, kde=True, stat="density", color='blue', edgecolor='black')

    plt.grid()

    # Set plot title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()  # Close the figure to avoid display if running in notebooks


def save_grouped_bar_chart(values, labels, title="Grouped Bar Chart with Scatter", figsize=(10, 6), xlabel="Groups", ylabel="Average Value", filename="grouped_bar_chart.png"):
    """
    Creates a bar chart with group averages, overlays scatter points for individual values, and saves the plot to a file.
    
    Args:
        values (list): List of values.
        labels (list): List of associated group labels.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure (width, height).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str): File name to save the plot (default: 'grouped_bar_chart.png').
    """
    # Create a DataFrame from values and labels
    data = pd.DataFrame({'Values': values, 'Labels': labels})
    
    # Calculate the average of values for each group
    group_means = data.groupby('Labels')['Values'].mean().reset_index()
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Plot bar chart for group averages with color coding by groups
    sns.barplot(x='Labels', y='Values', data=group_means, palette='deep', alpha=0.6, edgecolor='black')

    # Overlay scatter points for individual values
    sns.stripplot(x='Labels', y='Values', data=data, color='black', jitter=True, size=6, alpha=0.7)
    
    # Set plot title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()  # Close the figure to avoid display if running in notebooks

import numpy as np
import matplotlib.pyplot as plt

def save_boxplot_2(data, names, subgroups, filename, marker_labels, figsize=(15,4.5), title=None, ylabel=None, xlabel=None, xlim=None):
    """
    Creates and saves a color-coded boxplot for the given list of lists, where each group can belong to a subgroup.
    Each data point within a group is assigned a different marker style based on the provided labels.

    Parameters:
    - data: List of lists containing numerical data for each category.
    - names: List of strings corresponding to the names of each list in data.
    - subgroups: List of subgroup labels corresponding to each group in data.
    - filename: The name of the file where the plot will be saved.
    - marker_labels: List of labels for each unique data point within groups. The length should match the data length.
    """
    # Ensure data is numerical
    data = [np.array(group, dtype=float) for group in data]
    data_box = [np.array([x for x in g if not np.isnan(x)]) for g in data]

    # Ensure subgroups is a numpy array (categorical or strings are fine)
    subgroups = np.array(subgroups)
    
    # Ensure names are a list of strings
    names = list(map(str, names))

    # Define unique subgroups and assign colors to them
    unique_subgroups = list(set(subgroups))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_subgroups)))  # Color palette
    
    # Create a color map for subgroups
    color_map = {subgroup: color for subgroup, color in zip(unique_subgroups, colors)}
    
    # Define a list of marker styles to cycle through
    marker_styles = ["$S$","$R$","$G$","$M$","$P$","$D$","$N$"]
    #marker_styles = ["o", "^", "s", "D", "v", "p", "*"]
    markers = {label: marker_styles[i % len(marker_styles)] for i, label in enumerate(marker_labels)}
    
    # Create the boxplot
    plt.figure(figsize=figsize)  # Adjust the size of the figure if needed
    boxprops = dict(linewidth=2, alpha=0.65)
    
    boxplot = plt.boxplot(data_box, patch_artist=True, notch=True, boxprops=boxprops)
    
    # Color each box according to its subgroup
    for patch, subgroup in zip(boxplot['boxes'], subgroups):
        patch.set_facecolor(color_map[subgroup])
    
    # Scatter individual data points with dark red markers and different marker styles
    dark_red_color = "indianred"
    for i, group in enumerate(data, start=1):
        x_positions = np.random.normal(i, 0.2, size=len(group))
        
        # Use the corresponding marker for each label based on the `marker_labels` list
        for j, (x, y) in enumerate(zip(
            x_positions, group)):
            label = marker_labels[j % len(marker_labels)]
            marker_style = markers[label]
            if not np.isnan(y):
                plt.scatter(x, y, color=dark_red_color, marker=marker_style, s=40,
                        alpha=0.8, edgecolor='k', zorder=10)
    
    # Add labels
    plt.xticks(ticks=range(1, len(names) + 1), labels=names)
    
    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)
    
    if xlim:
        plt.xlim(xlim)

    if ylabel:
        plt.ylabel(ylabel)
    
    # Create the color legend for subgroups
    color_handles = [plt.Line2D([0], [0], color=color_map[subgroup], lw=4, label=subgroup) 
                     for subgroup in unique_subgroups]
    color_legend = plt.legend(handles=color_handles, loc="upper left", title="Subgroups", bbox_to_anchor=(1.05, 1))
    
    # Create a separate marker legend for `marker_labels`
    marker_handles = [plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=dark_red_color, markersize=10, label=label)
                      for label, marker in markers.items()]
    marker_legend = plt.legend(handles=marker_handles, loc="upper left", title="Marker Labels", bbox_to_anchor=(1.05, 0.7))
    
    # Add the color legend to the plot as a separate artist
    plt.gca().add_artist(color_legend)
    
    # Add grid and save figure
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
