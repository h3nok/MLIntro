import os
import sys

sys.path.append(r'..\\')
from correlation import ClassifierCorrelationPlots
from performance import ClassifierPerformanceReport
import database_interface as dbi
from providers.reportdata_provider import viNetResultDataProvider
from common.io.interactive import query_tag, query_config, query_customer, query_class_group

import argparse

parser = argparse.ArgumentParser(description='Report generator for viNet Neural Networks. any argument left blank. '
                                             'The CLI will walk you through what to select')
parser.add_argument('--customer_name', '-cn', default=None, help='The customer name, ex Vatenfall')
parser.add_argument('--config', '-c', default=None, help='Config')
parser.add_argument('--tag', '-t', default=None, help='Tag')
parser.add_argument('--classification_group', '-cg', default=None, help='The classification group')
parser.add_argument('--report_type', '-rt', default=None, type=int,
                    help='type of report, either classification report (0) or performance report (1)')
parser.add_argument('--limit', '-l', default=None, help='Number of frames to collect or "all"')
parser.add_argument('--output_dir', '-o', default=r'E:\viNet_RnD\Deployment\Metrics',
                    help='Output directory where report files are saved')
parser.add_argument('--db_config', default=r'C:\ProgramData\viNet\config\database.ini',
                    help="Database config file to create connection string from")

if __name__ == '__main__':
    args = parser.parse_args()
    customer = args.customer_name
    config = args.config
    tag = args.tag
    c_group = args.classification_group
    option = args.report_type
    limit = args.limit

    # DB Server
    server = dbi.PgsqlInterface()
    assert os.path.exists(
        args.db_config), \
        r'C:\ProgramData\viNet\config\database.ini Not found'
    server.connect(r'C:\ProgramData\viNet\config\database.ini')
    print("\n")

    # Customer
    customer = query_customer()

    # Config

    config = query_config(server.connection, customer)

    # Tag
    tag = query_tag(server.connection, customer)

    # classification group
    if not c_group:
        c_group = query_class_group(server)

    # report type
    reports = [ClassifierCorrelationPlots, ClassifierPerformanceReport]
    options = ["Correlation Report", "Performance Report"]
    reporter = None
    if option is not None and option >= len(reports):
        option = None

    while reporter is None:
        for i in range(len(options)):
            print(f"{i}. {options[i]}")
        print("\n")

        try:
            option = int(input("Select report type: "))
            reporter = reports[option]
        except (KeyError, IndexError, ValueError):
            print("Could not understand input. Insure input is of type int and in the range")
            reporter = None
        print("\n")

    # limit
    if limit is not None and (limit != 'all' or not isinstance(limit, int)):
        limit = None

    if limit is None:
        limit = input("Limit number of frames (int, press <enter > to use all frames: ")
        print("\n")
    if not limit:
        limit = 'all'
    # Collect Data
    print("collecting data ....")
    print(f"Report Type: {options[option]}")
    print(f"Customer: {customer.name}")
    print(f"Config: {config}")
    print(f"Tag: {tag}")
    print(f"Classification Group: {c_group}")
    print("\n")
    reports_dir = os.path.join(args.output_dir, customer.name)

    if option == 0:
        with viNetResultDataProvider(tag, config, classification_group=c_group,
                                     include_images=True) as dp:
            if limit.lower() != 'all':
                dp.set_limit(int(limit))
            dp.fetch_results()
            dp.to_csv(os.path.join(reports_dir, 'corr_data.csv'))
            dp.map_results()
            vinet_datum = dp.dataframe

            vinet_datum.vi.set_attributes(customer=customer)
    else:
        sproc = 'get_network_classification_results_on_grouped_frames'
        with viNetResultDataProvider(tag, config, classification_group=c_group,
                                     procedure=sproc) as dp:
            if limit.lower() != 'all':
                dp.set_limit(int(limit))
            dp.fetch_results()
            dp.to_csv(os.path.join(reports_dir, '{}_per_data.csv'.format(config)))
            vinet_datum = dp.dataframe
            vinet_datum.vi.set_attributes(customer=customer)

    print("Creating Report ....\n")
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    filename = os.path.join(reports_dir, f"{config}.pdf")
    reporter.create_complete_report(vinet_datum,
                                    save_loc=filename,
                                    appendix=False,
                                    colors=customer.colors)

    print(f"\nReport saved to: {filename}")
