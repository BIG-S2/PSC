#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import numpy as np
import pandas as pd

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exists

DESCRIPTION = 'Convert a final aggregated json file to an Excel spreadsheet.'


def _get_all_bundle_names(stats):
    bnames = set()

    for bundles in stats.itervalues():
        bnames |= set(bundles.keys())

    return list(bnames)


def _are_all_elements_scalars(bundle_stat):
    for v in bundle_stat.itervalues():
        if type(v) is not int and type(v) is not float:
            return False

    return True


def _get_metrics_names(stats):
    mnames = set()

    for bundles in stats.itervalues():
        for val in bundles.itervalues():
            mnames |= set(val.keys())

    return mnames


def _get_labels(stats):
    labels = set()

    for bundles in stats.itervalues():
        for lab in bundles.itervalues():
            if type(lab[lab.keys()[0]]) is dict:
                for vals in lab.itervalues():
                    labels |= set(vals.keys())
            else:
                labels |= set(lab.keys())

    return list(labels)


def _find_stat_name(stats):
    first_sub_stats = stats[stats.keys()[0]]
    first_bundle_stats = first_sub_stats[first_sub_stats.keys()[0]]

    return first_bundle_stats.keys()[0]


def _get_stats_parse_function(stats, stats_over_population):
    first_sub_stats = stats[stats.keys()[0]]
    first_bundle_stats = first_sub_stats[first_sub_stats.keys()[0]]
    first_bundle_substat = first_bundle_stats[first_bundle_stats.keys()[0]]

    if len(first_bundle_stats.keys()) == 1 and\
            _are_all_elements_scalars(first_bundle_stats):
        return _parse_scalar_stats
    elif len(first_bundle_stats.keys()) == 4 and \
            set(first_bundle_stats.keys()) == \
            set(['min_length', 'max_length', 'mean_length', 'std_length']):
        return _parse_lengths
    elif type(first_bundle_substat) is dict:
        sub_keys = first_bundle_substat.keys()
        if set(sub_keys) == set(['mean', 'std']):
            if stats_over_population:
                return _parse_per_label_population_stats
            else:
                return _parse_scalar_meanstd
        elif type(first_bundle_substat[sub_keys[0]]) is dict:
            return _parse_per_point_meanstd
        elif _are_all_elements_scalars(first_bundle_substat):
            return _parse_per_label_scalar

    raise IOError('Unable to recognize stats type!')


def _write_dataframes(dataframes, df_names, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for df, df_name in zip(dataframes, df_names):
            df.to_excel(writer, sheet_name=df_name)


def _parse_scalar_stats(stats, subs, bundles):
    stat_name = _find_stat_name(stats)

    nb_subs = len(subs)
    nb_bundles = len(bundles)

    stats_array = np.full((nb_subs, nb_bundles), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                stats_array[sub_id, bundle_id] = b_stat[stat_name]

    dataframes = [pd.DataFrame(data=stats_array,
                               index=subs,
                               columns=bundles)]
    df_names = [stat_name]

    return dataframes, df_names


def _parse_scalar_meanstd(stats, subs, bundles):
    metric_names = _get_metrics_names(stats)

    nb_subs = len(subs)
    nb_bundles = len(bundles)
    nb_metrics = len(metric_names)

    means = np.full((nb_subs, nb_bundles, nb_metrics), np.NaN)
    stddev = np.full((nb_subs, nb_bundles, nb_metrics), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            for metric_id, metric_name in enumerate(metric_names):
                b_stat = stats[sub_name].get(bundle_name)

                if b_stat is not None:
                    m_stat = b_stat.get(metric_name)

                    if m_stat is not None:
                        means[sub_id, bundle_id, metric_id] = m_stat['mean']
                        stddev[sub_id, bundle_id, metric_id] = m_stat['std']

    dataframes = []
    df_names = []

    for metric_id, metric_name in enumerate(metric_names):
        dataframes.append(pd.DataFrame(data=means[:, :, metric_id],
                                       index=subs, columns=bundles))
        df_names.append(metric_name + "_mean")

        dataframes.append(pd.DataFrame(data=stddev[:, :, metric_id],
                                       index=subs, columns=bundles))
        df_names.append(metric_name + "_std")

    return dataframes, df_names


def _parse_lengths(stats, subs, bundles):
    nb_subs = len(subs)
    nb_bundles = len(bundles)

    min_lengths = np.full((nb_subs, nb_bundles), np.NaN)
    max_lengths = np.full((nb_subs, nb_bundles), np.NaN)
    mean_lengths = np.full((nb_subs, nb_bundles), np.NaN)
    std_lengths = np.full((nb_subs, nb_bundles), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                min_lengths[sub_id, bundle_id] = b_stat['min_length']
                max_lengths[sub_id, bundle_id] = b_stat['max_length']
                mean_lengths[sub_id, bundle_id] = b_stat['mean_length']
                std_lengths[sub_id, bundle_id] = b_stat['std_length']

    dataframes = [pd.DataFrame(data=min_lengths,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=max_lengths,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=mean_lengths,
                               index=subs,
                               columns=bundles),
                  pd.DataFrame(data=std_lengths,
                               index=subs,
                               columns=bundles)]

    df_names = ["min_length", "max_length", "mean_length", "std_length"]

    return dataframes, df_names


def _parse_per_label_scalar(stats, subs, bundles):
    labels = _get_labels(stats)
    labels.sort()

    nb_subs = len(subs)
    nb_bundles = len(bundles)
    nb_labels = len(labels)

    stats_array = np.full((nb_subs, nb_bundles * nb_labels), np.NaN)
    column_names = []
    for bundle_name in bundles:
        column_names.extend(["{}_{}".format(bundle_name, label)
                             for label in labels])

    stat_name = _find_stat_name(stats)
    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name).get(stat_name)

            if b_stat is not None:
                for label_id, label in enumerate(labels):
                    label_stat = b_stat.get(label)

                    if label_stat is not None:
                        stats_array[sub_id,
                                    bundle_id * len(labels) + label_id] =\
                            label_stat

    dataframes = [pd.DataFrame(data=stats_array,
                               index=subs,
                               columns=column_names)]
    df_names = ['{}_per_label'.format(stat_name)]

    return dataframes, df_names


def _parse_per_point_meanstd(stats, subs, bundles):
    labels = _get_labels(stats)
    labels.sort()

    metric_names = _get_metrics_names(stats)

    nb_subs = len(subs)
    nb_bundles = len(bundles)
    nb_labels = len(labels)
    nb_metrics = len(metric_names)

    means = np.full((nb_subs, nb_bundles * nb_labels, nb_metrics), np.NaN)
    stddev = np.full((nb_subs, nb_bundles * nb_labels, nb_metrics), np.NaN)

    for sub_id, sub_name in enumerate(subs):
        for bundle_id, bundle_name in enumerate(bundles):
            b_stat = stats[sub_name].get(bundle_name)

            if b_stat is not None:
                for metric_id, metric_name in enumerate(metric_names):
                    m_stat = b_stat.get(metric_name)

                    if m_stat is not None:
                        for label_id, label in enumerate(labels):
                            label_stat = m_stat.get(label)

                            if label_stat is not None:
                                means[sub_id,
                                      bundle_id * len(labels) + label_id,
                                      metric_id] =\
                                    label_stat['mean']
                                stddev[sub_id,
                                       bundle_id * len(labels) + label_id,
                                       metric_id] =\
                                    label_stat['std']

    column_names = []
    for bundle_name in bundles:
        column_names.extend(["{}_{}".format(bundle_name, label)
                             for label in labels])

    dataframes = []
    df_names = []
    for metric_id, metric_name in enumerate(metric_names):
        dataframes.append(pd.DataFrame(data=means[:, :, metric_id],
                                       index=subs, columns=column_names))
        df_names.append(metric_name + "_mean")

        dataframes.append(pd.DataFrame(data=stddev[:, :, metric_id],
                                       index=subs, columns=column_names))
        df_names.append(metric_name + "_std")

    return dataframes, df_names


def _parse_per_label_population_stats(stats, bundles, metrics):
    labels = stats[bundles[0]][metrics[0]].keys()
    labels.sort()

    nb_bundles = len(bundles)
    nb_labels = len(labels)
    nb_metrics = len(metrics)

    means = np.full((nb_bundles * nb_labels, nb_metrics), np.NaN)
    stddev = np.full((nb_bundles * nb_labels, nb_metrics), np.NaN)

    for bundle_id, bundle_name in enumerate(bundles):
        b_stat = stats.get(bundle_name)

        if b_stat is not None:
            for metric_id, metric_name in enumerate(metrics):
                m_stat = b_stat.get(metric_name)

                if m_stat is not None:
                    for label_id, label in enumerate(labels):
                        label_stat = m_stat.get(label)

                        if label_stat is not None:
                            means[bundle_id * len(labels) + label_id,
                                  metric_id] =\
                                label_stat['mean']
                            stddev[bundle_id * len(labels) + label_id,
                                   metric_id] =\
                                label_stat['std']

    column_names = []
    for bundles_name in bundles:
        column_names.extend(["{}_{}".format(bundles_name, label)
                             for label in labels])

    dataframes = []
    df_names = []
    index = ['Population']
    for metric_id, metric_name in enumerate(metrics):
        dataframes.append(pd.DataFrame(data=np.array([means[:, metric_id]]),
                                       index=index,
                                       columns=column_names))
        df_names.append(metric_name + "_mean")

        dataframes.append(pd.DataFrame(data=np.array([stddev[:, metric_id]]),
                                       index=index,
                                       columns=column_names))
        df_names.append(metric_name + "_std")

    return dataframes, df_names


def _create_xlsx_from_json(json_path, xlsx_path,
                           sort_subs=True, sort_bundles=True,
                           ignored_bundles_fpath=None,
                           stats_over_population=False):
    with open(json_path, 'rb') as json_file:
        stats = json.load(json_file)

    subs = stats.keys()
    if sort_subs:
        subs.sort()

    bundle_names = _get_all_bundle_names(stats)
    if sort_bundles:
        bundle_names.sort()

    if ignored_bundles_fpath is not None:
        with open(ignored_bundles_fpath, 'r') as f:
            bundles_to_ignore = [l.strip() for l in f]
        bundle_names = filter(lambda name: name not in bundles_to_ignore,
                              bundle_names)

    cur_stats_func = _get_stats_parse_function(stats, stats_over_population)

    dataframes, df_names = cur_stats_func(stats, subs, bundle_names)

    if len(dataframes):
        _write_dataframes(dataframes, df_names, xlsx_path)


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('json_stats', action='store',
                   metavar='FILE', type=str,
                   help='File containing the json stats')

    p.add_argument('xlsx_stats', action='store',
                   metavar='FILE',  type=str,
                   help='Output Excel file for the stats.')

    p.add_argument('--no_sort_subs', action='store_false',
                   help='If set, subjects won\'t be sorted alphabetically.')

    p.add_argument('--no_sort_bundles', action='store_false',
                   help='If set, bundles won\'t be sorted alphabetically.')

    p.add_argument('--ignore_bundles', action='store', metavar='FILE',
                   help='Path to a text file containing a list of bundles '
                        'to ignore')

    p.add_argument('--stats_over_population', action='store_true',
                   help='If set, consider the input stats to be over an '
                        'entire population and not subject-based')

    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='overwrite output files')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    in_files = [args.json_stats]
    if args.ignore_bundles:
        in_files.append(args.ignore_bundles)

    assert_inputs_exist(parser, *in_files)
    assert_outputs_exists(parser, args, args.xlsx_stats)

    _create_xlsx_from_json(args.json_stats, args.xlsx_stats,
                           sort_subs=args.no_sort_subs,
                           sort_bundles=args.no_sort_bundles,
                           ignored_bundles_fpath=args.ignore_bundles,
                           stats_over_population=args.stats_over_population)


if __name__ == "__main__":
    main()
