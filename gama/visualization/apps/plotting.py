from typing import List

import pandas as pd
from plotly import graph_objects as go

from gama.logging.GamaReport import GamaReport


def individual_plot(report: GamaReport, x_axis: str, y_axis: str, mode: str):
    """

    :param report: report to pull data from
    :param x_axis: metric on the x-axis, column of report.evaluations
    :param y_axis: metric on the y-axis, column of report.evaluations
    :param mode: See `https://plot.ly/python/reference/#scatter-mode`
    :return:
        dash graph
    """
    return go.Scatter(
            name=f'{report.name}',
            x=report.evaluations[x_axis],
            y=report.evaluations[y_axis],
            text=[ind.short_name for ind in report.individuals.values()],
            mode=mode
        )


def aggregate_plot(reports_to_combine: List[GamaReport], x_axis: str, y_axis: str):
    """ Creates an aggregate plot over multiple reports by calculating the mean and std of `y_axis` by `x_axis`.

    :param reports_to_combine: reports of which to combine evaluations
    :param x_axis: column which is grouped by before aggregating `y_axis`
    :param y_axis: column over which to calculate the mean/std.
    :return:
        Three dash Scatter objects which respectively draw the lower bound, mean and upper bound.
    """
    concat_df = pd.concat([report.evaluations for report in reports_to_combine])
    concat_df = concat_df[concat_df[y_axis] != -float('inf')]
    agg_df = concat_df.groupby(by=x_axis).agg({y_axis: ['mean', 'std']}).reset_index()
    agg_df.columns = [x_axis, y_axis, 'std']
    upper_bound = go.Scatter(
        name=f'UB',
        x=agg_df[x_axis],
        y=agg_df[y_axis] + agg_df['std'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )

    mean_performance = go.Scatter(
        name=f'Mean',
        x=agg_df[x_axis],
        y=agg_df[y_axis],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )

    lower_bound = go.Scatter(
        name=f'LB',
        x=agg_df[x_axis],
        y=agg_df[y_axis] - agg_df['std'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0)
    )
    aggregate_data = [lower_bound, mean_performance, upper_bound]
    return aggregate_data