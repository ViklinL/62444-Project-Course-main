
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
from keplergl import KeplerGl
import pandas as pd

def kepler_map(df_trips, df_zones, time_window_hours=2, 
                      pickup_datetime_column='tpep_pickup_datetime', 
                      dropoff_datetime_column='tpep_dropoff_datetime'):
    """
    Creates a Kepler map for trips data with pickup and dropoff locations, and routes
    for the specified time window.
    
    Parameters:
    - df_trips: DataFrame containing trip data (yellow or green).
    - df_zones: DataFrame containing the zones (location ID, lat, lng).
    - time_window_hours: The time window in hours to filter trips (default is 2 hours).
    - pickup_datetime_column: Name of the pickup datetime column (default is 'tpep_pickup_datetime').
    - dropoff_datetime_column: Name of the dropoff datetime column (default is 'tpep_dropoff_datetime').
    
    Returns:
    - KeplerGl map object
    """
    # Choosing a time window to visualize the data
    start_time = df_trips[pickup_datetime_column].median()
    end_time = start_time + pd.Timedelta(hours=time_window_hours)

    # Filter trips data for trips within the time window
    df_filtered = df_trips[
        (df_trips[pickup_datetime_column] >= start_time) &
        (df_trips[pickup_datetime_column] <= end_time)
    ].copy()

    # Ordering zones after LocationID index
    df_zones_indexed = df_zones.set_index('LocationID')

    # Map pickup and dropoff lat/lng values
    df_filtered['pickup_lat'] = df_filtered['PULocationID'].map(df_zones_indexed['lat'])
    df_filtered['pickup_lng'] = df_filtered['PULocationID'].map(df_zones_indexed['lng'])
    df_filtered['dropoff_lat'] = df_filtered['DOLocationID'].map(df_zones_indexed['lat'])
    df_filtered['dropoff_lng'] = df_filtered['DOLocationID'].map(df_zones_indexed['lng'])

    # Create DataFrames for pickup and dropoff points
    df_pickup = pd.DataFrame({
        'lat': df_filtered['pickup_lat'],
        'lng': df_filtered['pickup_lng'],
        'type': 'pickup'  # Label as pickup
    })

    df_dropoff = pd.DataFrame({
        'lat': df_filtered['dropoff_lat'],
        'lng': df_filtered['dropoff_lng'],
        'type': 'dropoff'  # Label as dropoff
    })

    # Remove rows with missing values
    df_pickup = df_pickup.dropna(subset=['lat', 'lng'])
    df_dropoff = df_dropoff.dropna(subset=['lat', 'lng'])

    # Create DataFrame for routes between pickup and dropoff points
    df_routes = pd.DataFrame({
        'start_lat': df_filtered['pickup_lat'],
        'start_lng': df_filtered['pickup_lng'],
        'end_lat': df_filtered['dropoff_lat'],
        'end_lng': df_filtered['dropoff_lng']
    })
    routes_data = df_routes.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng']).copy()

    # Create a Kepler map
    map_1 = KeplerGl(height=600, width=800)
    map_1.add_data(data=df_pickup, name='pickup')
    map_1.add_data(data=df_dropoff, name='dropoff')
    map_1.add_data(data=routes_data, name='routes')

    # Return the Kepler map
    return map_1

####################################################################################################

def analyze_temporal_patterns(df, taxi_type):
    """
    Analyze temporal patterns for a given taxi dataset.
    Args:
        df (DataFrame): Taxi dataset.
        taxi_type (str): Type of taxi ("Yellow" or "Green").
    """
    # Ensure that correct pickup and dropoff datetime columns are used for (Yellow or Green) taxi
    if taxi_type.lower() == "yellow":
        pickup_datetime_column = 'tpep_pickup_datetime'
        dropoff_datetime_column = 'tpep_dropoff_datetime'
    elif taxi_type.lower() == "green":
        pickup_datetime_column = 'lpep_pickup_datetime'
        dropoff_datetime_column = 'lpep_dropoff_datetime'
    else:
        raise ValueError("Invalid taxi type. Use 'Yellow' or 'Green'.")

    # Convert pickup and dropoff times to datetime format
    df['pickup_datetime'] = pd.to_datetime(df[pickup_datetime_column])
    df['dropoff_datetime'] = pd.to_datetime(df[dropoff_datetime_column])
    
    # Extract temporal features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.day_name()
    df['month'] = df['pickup_datetime'].dt.month_name()
    
    # Number of rides by hour
    hourly_rides = df.groupby('hour').size()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=hourly_rides.index, y=hourly_rides.values, palette='coolwarm')
    plt.title(f"Number of Rides by Hour ({taxi_type} Taxi)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Rides")
    plt.show()
    
    # Number of rides by day of week
    day_rides = df.groupby('day_of_week').size()
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=day_rides.index, 
        y=day_rides.values, 
        palette='coolwarm', 
        order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    plt.title(f"Number of Rides by Day of Week ({taxi_type} Taxi)")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Rides")
    plt.show()
    
    # Number of rides by month
    month_rides = df.groupby('month').size()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=month_rides.index, y=month_rides.values, palette='coolwarm')
    plt.title(f"Number of Rides by Month ({taxi_type} Taxi)")
    plt.xlabel("Month")
    plt.ylabel("Number of Rides")
    plt.show()
    
    # Trip distance vs. hour
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='hour', y='trip_distance', data=df, palette='coolwarm')
    plt.title(f"Trip Distance by Hour of Day ({taxi_type} Taxi)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Trip Distance")
    plt.show()
    
    # Fare amount vs. hour
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='hour', y='fare_amount', data=df, palette='coolwarm')
    plt.title(f"Fare Amount by Hour of Day ({taxi_type} Taxi)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Fare Amount")
    plt.show()

####################################################################################################