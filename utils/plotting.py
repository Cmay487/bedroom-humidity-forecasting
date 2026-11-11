import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import title_params


def plot_time_series(
    dfs: list[pd.DataFrame],
    x_col: str,
    y_col: list[str],
    saved_name: str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    add_hlines: bool = False,
) -> None:
    """
    Plots a time series line plot.

    Parameters:
    dfs (list): List of DataFrames containing the data.
    x_col (str): Name of the column to be used for the x-axis (time).
    y_col (list[str]): List of column names to be used for the y-axis (values).
    title (str, optional): Title of the plot. Defaults to None.
    xlabel (str, optional): Label for the x-axis. Defaults to None.
    ylabel (str, optional): Label for the y-axis. Defaults to None.
    add_hlines (bool, optional): Whether to add horizontal lines. Defaults to False.
    """
 
    if len(y_col) > 1:
        fig, axes = plt.subplots(len(y_col), 1, figsize=(12, 8))

        # plot the subplots
        plot_subplots(axes, y_col, x_col, dfs[0])

        # clean plot
        plot_cleanup(axes, title, xlabel, ylabel)

    else:
        fig = plt.figure(figsize=(12, 8))

        # plt a single plot
        plot_single_plot(dfs, x_col, y_col[0])

        # clean plot
        plot_cleanup(None, title, xlabel, ylabel)


    if add_hlines:
        # add horizontal lines
        plot_hlines(fig, [40, 60])

    # save the image in plts
    save_image(saved_name, fig)

    plt.legend()
    # show result
    plt.show()

def plot_subplots(
        axes:object, 
        y_col: list[str], 
        x_col: str, 
        df: pd.DataFrame,
        ) -> None:
    """
    Plots subplots for the given y_col and x_col.
    
    Parameters:
    axes (object): The axes object to plot on.
    y_col (list[str]): List of column names to be used for the y-axis (values).
    x_col (str): Name of the column to be used for the x-axis (time).
    df (pd.DataFrame): The DataFrame containing the data.
    """
    for idx, y in enumerate(y_col):
        sns.lineplot(
            data=df, 
            x=x_col, 
            y=y,
            alpha=0.5,
            color='blue' if idx == 1 else 'green',
            ax=axes[idx]
        )

def plot_single_plot(
        dfs: list[pd.DataFrame], 
        x_col: str, 
        y_col: list[str]
        ) -> None:
    """
    Plots a single line plot for the given y_col and x_col.

    Parameters:
    dfs (list): List of DataFrames containing the data.
    x_col (str): Name of the column to be used for the x-axis (time).
    y_col (list[str]): List of column names to be used for the y-axis (values).
    """
    if len(dfs) == 2:
        labels = ["Window Sensor", "Door Sensor"]
    else:
        labels = [y_col]
    # because we are comparing 2 dfs, use a counter to determine the color of the line
    for idx, df in enumerate(dfs):
        sns.lineplot(
            data=df, 
            x=x_col, 
            y=y_col[idx] if isinstance(y_col, list) else y_col,
            alpha=0.5,
            label=labels[idx],
            color='blue' if idx == 1 else 'green',
        )

def plot_hlines(fig: object,y_values: list[float]) -> None:
    """
    Adds horizontal lines to the plot.

    Parameters:
    fig (object): The figure object to add the lines to.
    y_values (list[float]): List of y-values where the lines should be drawn.
    """

    # loop through the axes and add the horizontal lines
    for ax in fig.axes:
        for y in y_values:
            ax.axhline(
                y=y,
                color='grey',
                linestyle='--',
                alpha=0.5,
                linewidth=2.5,
            )
            
def plot_cleanup(
        axes: object | None = None, 
        title: str | None = None, 
        xlabel: str | None = None, 
        ylabel: str | None = None
        ) -> None:
    """
    Cleans up the plot by setting the title, xlabel, ylabel, and rotating the x-ticks.

    Parameters:
    axes (object, optional): The axes object to clean up. Defaults to None.
    title (str, optional): Title of the plot. Defaults to None.
    xlabel (str, optional): Label for the x-axis. Defaults to None.
    ylabel (str, optional): Label for the y-axis. Defaults to None.

    """    
    # add suptitle
    plt.suptitle(title, **title_params)
    if axes is not None:     
        for ax in axes:
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            # remove gridlines    
            ax.grid(False)
    else:
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        # remove gridlines
        plt.grid(False)

    # despine
    sns.despine()

def plot_histogram(
    dfs: list[pd.DataFrame],
    x_col: str,
    saved_name: str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    bins: int | None = None,
    
):
    """
    Plots a histogram.

    Parameters:
    dfs (list): List of DataFrames containing the data.
    x_col (str): Name of the column to be used for the x-axis (time).
    y_col (str): Name of the column to be used for the y-axis (values).
    title (str, optional): Title of the plot. Defaults to None.
    xlabel (str, optional): Label for the x-axis. Defaults to None.
    ylabel (str, optional): Label for the y-axis. Defaults to None.
    """

    # print the stats
    
    
    fig = plt.figure(figsize=(12, 8))

    # because we are always comparing 2 dfs, use a counter to determine the color of the line
    for i, df in enumerate(dfs):

        print(f"{df[x_col].describe()} \n", end="\n")
        # now review skewness and kurtosis

        print(f"Skewness: {df[x_col].skew()}")
        print(f"Kurtosis: {df[x_col].kurt()}")

        sns.histplot(
            data=df, 
            x=x_col, 
            kde=True,
            alpha=0.5,
            color='blue' if i == 1 else 'green',
            bins=bins
        )

    plot_cleanup(None, title, xlabel, ylabel)

    #save the image
    save_image(saved_name, fig)

def plot_relationships(
    df: pd.DataFrame,
    x_col: str,
    y_col: list[str],
    saved_name : str,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    rows : int | None = None
):  
    """
    
    """
    plots_rows = 2 if rows is None else rows
    fig, axes = plt.subplots(plots_rows, 1, figsize=(10, 8))
    
    # print the correlations
    print(df.corr(), end="\n")

    if plots_rows >= 2:
        for i, feature in enumerate(x_col):
            sns.regplot(
                data=df,
                x=feature,
                y=y_col,
                fit_reg=True,
                ci=95,
                ax=axes[i]
            )
    else:
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            fit_reg=True,
            ci=95
        )

    plot_cleanup(axes, title, xlabel, ylabel)


    # save the image
    save_image(saved_name, fig)


def save_image(file_name:str, fig:object):
    
    fig.savefig(f"plots/{file_name}.png", dpi=300, bbox_inches="tight")
