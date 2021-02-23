# although this is greyed out this is still a required import
from pprint import pprint

from providers.reportdata_provider import viNetResultDataProvider
from core.analytics.reporting.csv_generator import CSVGenerator
from common.db.stored_procedures import StoredProcedures as sp
import database_interface as dbi

from functools import reduce

import os

import regex as re


CSV_ROOT_PATH = r'C:\viNet_RnD\Selection'
SERVER_CONNECTION_FILE = r'C:\ProgramData\viNet\config\database.ini'


def calc_candidate_selection_metrics_for_tag_to_file(configs: list, tag, cgroup, protected_classes: list, csv_dir):
    file_name = re.sub(r'[\\/:*?"<>| ]', '_', tag) + '.csv'

    csv_file = os.path.join(csv_dir, file_name)
    for config in configs:
        calc_candidate_selection_metrics_for_config_to_file(config, tag, cgroup, protected_classes, csv_file)
    return csv_file


def calc_candidate_selection_metrics_for_config_to_file(config, tag, cgroup, protected_classes: list, csv_file):
    sproc = 'get_network_classification_results_on_grouped_frames'

    with viNetResultDataProvider(tag, config,
                                 classification_group=cgroup,
                                 procedure=sproc) as dp:
        dp.fetch_results()

        try:
            vinet_datum = dp.dataframe

            vinet_datum.vi.binarize(protected_classes)

            try:
                metrics = vinet_datum.vi.calc_metrics('all')
                acc = metrics['accuracy']
                bacc = metrics['binary_accuracy']
                fnr = metrics['fnr']
                fpr = metrics['fpr']
                tnr = metrics['tnr']
                tpr = metrics['tpr']
                err = None
            except BaseException as e:  # the main reason this will fail is if `protected_classes == []`, but that's fine.
                metrics = vinet_datum.vi.calc_metrics(['accuracy'])
                acc = metrics['accuracy']
                err = e

            # add to csv
            with CSVGenerator(csv_file, allow_append=True, entries_str='configs') as csvf:
                entry_name = config
                entry_data = dict()
                entry_data['accuracy'] = acc
                if err is None:
                    entry_data['binary_accuracy'] = bacc
                    entry_data['fnr'] = fnr
                    entry_data['fpr'] = fpr
                    entry_data['tnr'] = tnr
                    entry_data['tpr'] = tpr
                else:
                    entry_data['error'] = err

                csvf.write_line(entry_name, entry_data)
        except BaseException as e:
            print('failed to calc_candidate_selection_metrics_to_file, error: ', e)


def get_input_option(options, print_val, csv=False):
    while True:
        print(print_val)
        try:
            for i, op_name in enumerate(options):
                print("{}: {}".format(i, op_name))
            if csv:
                return [options[int(n)] for n in input().split(',')]
            else:
                return options[int(input())]
        except (ValueError, IndexError):
            print('please try again')


def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.

    :param question: is a string that is presented to the user.
    :param default: is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    :return: The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "yeah": True, "yup": True, 'certainly': True, 'uh-huh': True,
             "no": False, "n": False, "nope": False, "nay": False, "nah": False}
    if default is None:
        prompt = " [y/n] "
    elif default.lower() not in valid:
        raise ValueError("invalid default answer: '%s'" % default)
    elif valid[default.lower()]:
        prompt = " [Y/n] "
    elif not valid[default.lower()]:
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        choice = input(question + prompt).lower()
        if default is not None and choice == '':
            return valid[default.lower()]
        elif choice in valid:
            return valid[choice.lower()]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


def get_user_input_for_tag(tag, configs, class_groups, server):
    while True:
        # an input could be `vattenfall.4.class,120,121` which will get all configs that match `vattenfall.4.class` and
        # configs 120 and 121
        print("The tag is `{}`, please enter keyword (use regex or numbers comma separated not case sensitive) to select configs "
              "(leave blank to skip):".format(tag))
        this_keyword = input()
        this_cgroup = None
        this_protected_classes = []
        if this_keyword is None or this_keyword == '':
            break

        nums = [int(str_seg) for str_seg in this_keyword.split(',') if str_seg.isnumeric()]
        not_nums = [str_seg for str_seg in this_keyword.split(',') if not str_seg.isnumeric()]

        configs_by_nums = [configs[num] for num in nums]

        configs_by_not_nums = reduce(lambda a, b:
                                        a + list(filter(lambda c: re.search(b, c, re.IGNORECASE) is not None, configs)),
                                     not_nums, [])

        filtered_configs = list(set(configs_by_nums + configs_by_not_nums))

        #filtered_configs = list(filter(lambda c: re.search(this_keyword, c, re.IGNORECASE) is not None, configs))
        print(len(filtered_configs), 'configs were found:')
        pprint(filtered_configs)

        if len(filtered_configs) > 0:
            group_list = list(class_groups['name'])
            this_cgroup = get_input_option(group_list, 'What is the classification group (enter number):')

            classes = list(set(sp.get_network_truth_mapping(server.database_hook, this_cgroup)['group_name']))
            this_protected_classes = get_input_option(classes,
                                                      'What are the protected classes (comma separated numbers):',
                                                      csv=True)

        if not query_yes_no('Redo Tag?', 'no'):
            break

        print('redoing tag')

    return this_keyword, this_cgroup, this_protected_classes


if __name__ == '__main__':
    """
    Will create a CSV file of metrics from configs for each tag.
    
    You are asked to select all of the configs you want to look at for each tag.
    """
    server = dbi.PgsqlInterface()
    server.connect(SERVER_CONNECTION_FILE)
    configs = sp.vinet_configs(server.connection)
    tags, _ = sp.dataset_tags(server.connection)
    class_groups = sp.vinet_classification_groups(server.connection)

    print('There are', len(configs), 'configs:')
    for i, config in enumerate(configs):
        print(f"{i}: {config}")

    if not os.path.exists(CSV_ROOT_PATH):
        os.makedirs(CSV_ROOT_PATH)

    tag_data_dict = dict()

    # Get keywords for tags
    print("There are", len(tags), "tags.")
    for tag in tags:
        tag_data_dict[tag] = get_user_input_for_tag(tag, configs, class_groups, server)

    print("working on calculating metrics; this may take a while.")
    print("output dir:", CSV_ROOT_PATH)

    for tag in tags:
        keyword, cgroup, protected_classes = tag_data_dict[tag]  # '2.7_vattenfall_4_class'
        if keyword is None or keyword == '':
            continue

        filtered_configs = list(filter(lambda c: re.search(keyword, c, re.IGNORECASE) is not None, configs))

        if len(filtered_configs) == 0:
            continue

        out_file = calc_candidate_selection_metrics_for_tag_to_file(filtered_configs, tag, cgroup, protected_classes,
                                                                    CSV_ROOT_PATH)

        print("Finished writing metrics for", tag, "to file:", out_file)
