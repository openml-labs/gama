from typing import List

import pandas as pd
from plotly import graph_objects as go

from gama.logging.GamaReport import GamaReport


def plot_preset_graph(reports: List[GamaReport], preset: str):
    if reports == []:
        return {}

    plots = []
    layout = {}
    first_metric = f'{reports[0].metrics[0]}'
    first_metric_max = f'{first_metric}_cummax'

    if preset == 'best_over_n':
        plots = [individual_plot(report, 'n', first_metric_max, 'lines')
                 for report in reports]
        layout = dict(
            title='Best score by iteration',
            xaxis=dict(title='n'),
            yaxis=dict(title=f'max {first_metric}'),
            hovermode='closest'
        )
    elif preset == 'best_over_time':
        plots = [individual_plot(report, 'relative_end', first_metric_max, 'lines')
                 for report in reports]
        layout = dict(
            title=f'Best score over time',
            xaxis=dict(title='time (s)'),
            yaxis=dict(title=f'max {first_metric}'),
            hovermode='closest'
        )
    elif preset == 'size_vs_metric':
        plots = [individual_plot(report, first_metric, 'length', 'markers')
                 for report in reports]
        layout = dict(
            title=f'Size vs {first_metric}',
            xaxis=dict(title=first_metric),
            yaxis=dict(title='pipeline length'),
            hovermode='closest'
        )
    elif preset == 'number_pipeline_by_size':
        for report in reports:
            size_counts = report.evaluations.length.value_counts()
            size_ratio = size_counts / len(report.individuals)
            plots.append(go.Bar(
                x=size_ratio.index.values,
                y=size_ratio.values,
                name=report.name)
            )
        layout = dict(
            title=f'Number of pipelines by size',
            xaxis=dict(title='pipeline length'),
            yaxis=dict(title='pipeline count')
        )
    elif preset == 'number_pipeline_by_learner':
        for report in reports:
            main_learners = [str(ind.main_node._primitive) for ind in report.individuals.values()]
            learner_counts = pd.Series(main_learners).value_counts()
            learner_ratio = learner_counts / len(report.individuals)
            plots.append(go.Bar(
                x=learner_ratio.index.values,
                y=learner_ratio.values,
                name=report.name)
            )
        layout = dict(
            title=f'Number of pipelines by size',
            xaxis=dict(title='pipeline length'),
            yaxis=dict(title='pipeline count')
        )
    return {
        'data': plots,
        'layout': layout
    }


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