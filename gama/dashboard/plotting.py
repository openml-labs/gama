from typing import List, Optional

import pandas as pd
from plotly import graph_objects as go

from gama.logging.GamaReport import GamaReport


def plot_preset_graph(reports: List[GamaReport], preset: Optional[str]):
    if reports == []:
        return {}

    plots = []
    layout = {}
    first_metric = f"{reports[0].metrics[0]}"
    first_metric_max = f"{first_metric}_cummax"

    if preset == "best_over_n":
        # if aggregate == "separate-line":
        plots = [
            individual_plot(report, "n", first_metric_max, "lines")
            for report in reports
        ]
        # elif aggregate == "aggregate":
        #     plots = aggregate_plot(aggregate_df, "n", first_metric_max)
        layout = dict(
            title="Best score by iteration",
            xaxis=dict(title="n"),
            yaxis=dict(title=f"max {first_metric}"),
            hovermode="closest",
        )
    elif preset == "best_over_time":
        # if aggregate == "separate-line":
        plots = [
            individual_plot(report, "relative_end", first_metric_max, "lines")
            for report in reports
        ]
        # elif aggregate == "aggregate":
        #     plots = aggregate_best_over_time(aggregate_df, first_metric_max)
        layout = dict(
            title=f"Best score over time",
            xaxis=dict(title="time (s)"),
            yaxis=dict(title=f"max {first_metric}"),
            hovermode="closest",
        )
    elif preset == "size_vs_metric":
        # if aggregate == "separate-line":
        plots = [
            individual_plot(report, first_metric, "length", "markers")
            for report in reports
        ]
        # elif aggregate == "aggregate":
        #     plots = []
        #     for method in aggregate_df.search_method.unique():
        #         method_df = aggregate_df[aggregate_df.search_method == method]
        #         plots.append(
        #             go.Scatter(
        #                 x=method_df[first_metric],
        #                 y=method_df.length,
        #                 mode="markers",
        #                 name=method,
        #             )
        #         )
        layout = dict(
            title=f"Size vs {first_metric}",
            xaxis=dict(title=first_metric),
            yaxis=dict(title="pipeline length"),
            hovermode="closest",
        )
    elif preset == "number_pipeline_by_size":
        # if aggregate == "separate-line":
        for report in reports:
            size_counts = report.evaluations.length.value_counts()
            size_ratio = size_counts / len(report.individuals)
            plots.append(
                go.Bar(x=size_ratio.index.values, y=size_ratio.values, name=report.name)
            )
        # elif aggregate == "aggregate":
        #     for method in aggregate_df.search_method.unique():
        #        results_for_method = aggregate_df[aggregate_df.search_method == method]
        #        size_counts = results_for_method.length.value_counts()
        #        size_ratio = size_counts / len(results_for_method)
        #        plots.append(
        #            go.Bar(x=size_ratio.index.values, y=size_ratio.values, name=method)
        #        )
        layout = dict(
            title=f"Ratio of pipelines by size",
            xaxis=dict(title="pipeline length"),
            yaxis=dict(title="pipeline count"),
        )
    elif preset == "number_pipeline_by_learner":
        for report in reports:
            main_learners = [
                str(ind.main_node._primitive) for ind in report.individuals.values()
            ]
            learner_counts = pd.Series(main_learners).value_counts()
            learner_ratio = learner_counts / len(report.individuals)
            plots.append(
                go.Bar(
                    x=learner_ratio.index.values,
                    y=learner_ratio.values,
                    name=report.name,
                )
            )
        layout = dict(
            title=f"Ratio of pipelines by learner",
            xaxis=dict(title="pipeline length"),
            yaxis=dict(title="learner"),
        )
    elif preset == "evaluation_times_dist":
        # if aggregate == "separate-line":
        for report in reports:
            time_s = report.evaluations.duration.dt.total_seconds()
            plots.append(go.Histogram(x=time_s, name=report.name))
        # elif aggregate == "aggregate":
        #     for method in aggregate_df.search_method.unique():
        #         time_s = aggregate_df[
        #             aggregate_df.search_method == method
        #         ].duration.dt.total_seconds()
        #         plots.append(go.Histogram(x=time_s, name=method))
        layout = dict(
            title=f"Pipeline Evaluation Times",
            xaxis=dict(title="duration (s)"),
            yaxis=dict(title="count"),
        )
    elif preset == "n_by_rung":
        for report in reports:
            if report.search_method == "AsynchronousSuccessiveHalving":
                count_by_rung = (
                    report.evaluations.groupby(by="rung").n.count().reset_index()
                )
                plots.append(
                    go.Bar(x=count_by_rung.rung, y=count_by_rung.n, name=report.name)
                )
        layout = dict(
            title=f"#Evaluations by Rung",
            xaxis=dict(title="rung"),
            yaxis=dict(title="count"),
        )
    elif preset == "time_by_rung":
        for report in reports:
            if report.search_method == "AsynchronousSuccessiveHalving":
                duration_by_rung = (
                    report.evaluations.groupby(by="rung").duration.sum().reset_index()
                )
                duration_by_rung.duration = duration_by_rung.duration.dt.total_seconds()
                plots.append(
                    go.Bar(
                        x=duration_by_rung.rung,
                        y=duration_by_rung.duration,
                        name=report.name,
                    )
                )
        layout = dict(
            title=f"Time spent per Rung",
            xaxis=dict(title="rung"),
            yaxis=dict(title="time (s)"),
        )
    return {"data": plots, "layout": layout}


def individual_plot(report: GamaReport, x_axis: str, y_axis: str, mode: str):
    """

    :param report: report to pull data from
    :param x_axis: metric on the x-axis, column of report.evaluations
    :param y_axis: metric on the y-axis, column of report.evaluations
    :param mode: See `https://plot.ly/python/reference/#scatter-mode`
    :return:
        dash graph
    """
    print(len(report.evaluations[x_axis]))
    print(len(report.evaluations[y_axis]))
    return go.Scatter(
        name=f"{report.name}",
        x=report.evaluations[x_axis],
        y=report.evaluations[y_axis],
        text=[ind.short_name for ind in report.individuals.values()],
        mode=mode,
    )


def aggregate_best_over_time(aggregate: pd.DataFrame, y_axis: str):
    # By method, find the best score per trace at each point in time.
    # I'm sure there is a better way to do this...
    colors = {
        0: "rgba(255, 0, 0, {a})",
        1: "rgba(0, 255, 0, {a})",
        2: "rgba(0, 0, 255, {a})",
    }
    soft_colors = {i: color.format(a=0.2) for i, color in colors.items()}
    hard_colors = {i: color.format(a=1.0) for i, color in colors.items()}

    aggregate_data: List[go.Scatter] = []
    aggregate = aggregate[aggregate[y_axis] != -float("inf")]
    for color_no, method in enumerate(aggregate.search_method.unique()):
        method_agg = aggregate[aggregate.search_method == method]
        all_end_times = method_agg.relative_end.unique()
        end_time_series = pd.Series(sorted(all_end_times), name="relative_end")

        scores_over_time = pd.DataFrame(end_time_series)
        for log_no in method_agg.log_no.unique():
            log_results = method_agg[method_agg.log_no == log_no]
            best_for_log = pd.merge_asof(
                end_time_series,
                log_results[["relative_end", y_axis]],
                on="relative_end",
            )
            scores_over_time[f"log{log_no}"] = best_for_log[y_axis]
        scores_over_time = scores_over_time.set_index("relative_end")
        mean_scores = scores_over_time.mean(axis=1)
        std_scores = scores_over_time.std(axis=1)
        scores_over_time["mean"] = mean_scores
        scores_over_time["std"] = std_scores

        upper_bound = go.Scatter(
            x=scores_over_time.index,
            y=scores_over_time["mean"] + scores_over_time["std"],
            mode="lines",
            marker=dict(color=soft_colors[color_no]),
            line=dict(width=0),
            fillcolor=soft_colors[color_no],
            fill="tonexty",
            showlegend=False,
        )

        mean_performance = go.Scatter(
            name=method,
            x=scores_over_time.index,
            y=scores_over_time["mean"],
            mode="lines",
            line=dict(color=hard_colors[color_no]),
            fillcolor=soft_colors[color_no],
            fill="tonexty",
        )

        lower_bound = go.Scatter(
            x=scores_over_time.index,
            y=scores_over_time["mean"] - scores_over_time["std"],
            mode="lines",
            marker=dict(color=soft_colors[color_no]),
            line=dict(width=0),
            showlegend=False,
        )
        aggregate_data += [lower_bound, mean_performance, upper_bound]
    return aggregate_data


def aggregate_plot(aggregate: pd.DataFrame, x_axis: str, y_axis: str):
    """ Create an aggregate plot over multiple reports.

     Aggregates the mean and std of `y_axis` by `x_axis`.

    :param aggregate: dataframe with all evaluations
    :param x_axis: column which is grouped by before aggregating `y_axis`
    :param y_axis: column over which to calculate the mean/std.
    :return:
        Three dash Scatter objects which respectively draw  the lower bound, mean and
        upper bound.
    """
    colors = {
        0: "rgba(255, 0, 0, {a})",
        1: "rgba(0, 255, 0, {a})",
        2: "rgba(0, 0, 255, {a})",
    }
    soft_colors = {i: color.format(a=0.2) for i, color in colors.items()}
    hard_colors = {i: color.format(a=1.0) for i, color in colors.items()}

    # concat_df = pd.concat([report.evaluations for report in reports_to_combine])
    # concat_df = concat_df[concat_df[y_axis] != -float('inf')]
    # agg_df = concat_df.groupby(by=x_axis).agg({y_axis: ['mean', 'std']}).reset_index()
    # agg_df.columns = [x_axis, y_axis, 'std']
    aggregate_data: List[go.Scatter] = []
    aggregate = aggregate[aggregate[y_axis] != -float("inf")]
    for color_no, method in enumerate(aggregate.search_method.unique()):
        agg_for_method = aggregate[aggregate.search_method == method]
        agg_df = (
            agg_for_method.groupby(by=x_axis)
            .agg({y_axis: ["mean", "std"]})
            .reset_index()
        )
        agg_df.columns = [x_axis, y_axis, "std"]

        upper_bound = go.Scatter(
            x=agg_df[x_axis],
            y=agg_df[y_axis] + agg_df["std"],
            mode="lines",
            marker=dict(color=soft_colors[color_no]),
            line=dict(width=0),
            fillcolor=soft_colors[color_no],
            fill="tonexty",
            showlegend=False,
        )

        mean_performance = go.Scatter(
            name=method,
            x=agg_df[x_axis],
            y=agg_df[y_axis],
            mode="lines",
            line=dict(color=hard_colors[color_no]),
            fillcolor=soft_colors[color_no],
            fill="tonexty",
        )

        lower_bound = go.Scatter(
            x=agg_df[x_axis],
            y=agg_df[y_axis] - agg_df["std"],
            mode="lines",
            marker=dict(color=soft_colors[color_no]),
            line=dict(width=0),
            showlegend=False,
        )
        aggregate_data += [lower_bound, mean_performance, upper_bound]
    return aggregate_data
