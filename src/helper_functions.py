
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
    df_outliers_filtered = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]
    

    return df_outliers_filtered

####################################################################################################

import matplotlib.pyplot as plt

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

    # Plot the histogram of the distribution
    plt.hist(df[col_name])
    plt.title(f'Distribution of {col_name} for the {car_colour} car')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.show()
    
    return None