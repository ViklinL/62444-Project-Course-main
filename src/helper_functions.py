
def  filter_outliers(df, col_name):

    """
    Function to filter outliers from the distribution of a given column
    parameters:
    -----------
    df: pandas dataframe

    col_name: name of the column to filter outliers from

    returns: 
    --------
    df_outliers_filtered: pandas dataframe with outliers removed from the distribution
    
    """

    # Calculate the 25 & 75 percentile range of the distubution
    Q1 = df[col_name].quantile(0.25)  
    Q3 = df[col_name].quantile(0.75)  

    # Calculate the interquartile range for the distribution
    IQR = Q3 - Q1                               

    # Define lower and upper bounds for the distribution
    upper_bound = Q3 + 1.5 * IQR

    # Lower bound is defined as 0, for the fare amount cannot be negative
    lower_bound = 0

    # Filter and remove outliers from the distribution
    df_outliers_filtered = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)].reset_index(drop=True)
    

    return df_outliers_filtered

####################################################################################################

import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(df, col_name, car_colour):
    """
    Function to plot the histogram of a given column
    parameters:
    -----------
    df: pandas dataframe

    col_name: name of the column to plot the histogram for
    
    car_colour: color of the car (used in the plot title)

    returns: 
    --------
    None
    """

    plt.figure(figsize=(8, 5))

    if col_name == 'passenger_count':
        # Determine integer-aligned bins
        min_value = int(df[col_name].min())
        max_value = int(df[col_name].max())
        
        bins = range(min_value, max_value + 2)
        
        # Plot histogram aligned to the left edge of each integer bin
        plt.hist(df[col_name], bins=bins, edgecolor='black', align='left')
        
        # Force the x-axis to show integer ticks
        plt.xticks(np.arange(min_value, max_value + 1, 1))
    else:
        # For continuous or non-integer columns, use 100 bins
        plt.hist(df[col_name], bins=100, edgecolor='black')
    
    # Add labels and title
    plt.title(f'Distribution of {col_name} for the {car_colour} car')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    
    plt.show()
    return None

####################################################################################################

def plot_scatter(df1, col1, df2, col2, title, xlabel, ylabel, alpha=0.5):
    """
    Function to plot a scatter plot of two columns from two aligned DataFrames.
    
    Parameters:
    -----------
    df1: pandas dataframe
        First DataFrame (for the x-axis data)
    col1: str
        Column name in the first DataFrame for the x-axis values
    df2: pandas dataframe
        Second DataFrame (for the y-axis data)
    col2: str
        Column name in the second DataFrame for the y-axis values
    title: str
        Title for the scatter plot
    xlabel: str
        Label for the x-axis
    ylabel: str
        Label for the y-axis
    alpha: float, optional
        Transparency level for scatter points (default is 0.5)

    Returns: 
    --------
    None
    """
    # Align the two DataFrames to ensure matching rows
    aligned_df1, aligned_df2 = df1.align(df2, join='inner')
    
    # Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(aligned_df1[col1], aligned_df2[col2], alpha=alpha)
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()
    return None

####################################################################################################

def sample_data_randomly(df1, df2, sample_size):
    """
    Randomly samples two aligned DataFrames to reduce the size of the data.

    parameters:
    -----------
    df1: pandas dataframe
        First DataFrame to sample from
    df2: pandas dataframe
        Second DataFrame to sample from
    sample_size: int
        Number of rows to sample

    returns:
    --------
    sampled_df1, sampled_df2: pandas dataframes
        Randomly sampled DataFrames
    """
    # Align the DataFrames to ensure they have the same indices
    aligned_df1, aligned_df2 = df1.align(df2, join='inner')
    
    # Perform random sampling
    sampled_indices = aligned_df1.sample(n=sample_size, random_state=42).index
    sampled_df1 = aligned_df1.loc[sampled_indices]
    sampled_df2 = aligned_df2.loc[sampled_indices]
    
    return sampled_df1, sampled_df2

####################################################################################################

import seaborn as sns

def plot_violin(df, x_col, y_col, title, xlabel, ylabel, color='skyblue'):
    """
    Function to create a violin plot for a given DataFrame and columns.
    
    Parameters:
    -----------
    df: pandas DataFrame
        The DataFrame containing the data.
    x_col: str
        The name of the column for the x-axis.
    y_col: str
        The name of the column for the y-axis.
    title: str
        The title of the plot.
    xlabel: str
        The label for the x-axis.
    ylabel: str
        The label for the y-axis.
    color: str, optional
        The color of the plot (default is 'skyblue').

    Returns:
    --------
    None
    """
    sns.violinplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        inner='quartile',  # Adds quartile lines to the plot
        #inner=None,
        color=color
    )

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.ylim(0, df[y_col].max())
    plt.show()

    return None

####################################################################################################


