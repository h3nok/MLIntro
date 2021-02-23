import os
import argparse
from block_timer.timer import Timer
from customer.customers import CustomerObjectMap
from dataset.generator.tfrecord_generator import TrainingSetGenerator
from dataset.providers.trainingset_provider import TrainingSetProvider
from dataset.tfrecord_explorer import DatasetExplorer as DataExp

parser = argparse.ArgumentParser(description='Dataset CLI')
parser.add_argument('--customer', default='GWA', help="Customer name")
parser.add_argument('--batch_size', default=100000, type=int, help="Number of frames per batch")
parser.add_argument('--limit', default=None, type=int, help="Number of frames per batch")
parser.add_argument('--augment', default=None, help="A dictionary of species and augmentation cmd")
parser.add_argument('--query', default='gwa.sql',
                    help="A sql query file or a script. If not supplied default customer query will be used")
parser.add_argument('--output_dir', default=r"E:\viNet_RnD\Datasets",
                    help="Location where the tfrecords are written to")
parser.add_argument('--dataset_version', default='GWA-V5', help="Dataset version used to isolate the data on "
                                                                       "disc")

if __name__ == '__main__':
    os.chdir(r"C:\svn\viAi\sql\Scripts")
    args = parser.parse_args()

    customer = CustomerObjectMap[args.customer]()
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, customer.name, args.dataset_version)

    batch_counter = 0
    record_counter = 0
    query_file = args.query or customer.default_query()

    if args.augment:
        augment = eval(args.augment)
    else:
        augment = {'Crow-Or-Raven': 'x5',
                   'Gull': 'x6',
                   'Buzzard': 'x9',
                   'White-Tailed-Eagle': 'x6',
                   'Golden-Eagle-Or-Kite': 'x9',
                   'Other-Avian-Gotlan-V2': 'x5'
                } # dah

    with Timer(title="TrainingSet Generator") as t:
        with TrainingSetProvider(query=query_file, batch_size=int(args.batch_size),
                                 limit=args.limit) as tsp:
            # num_records = tsp.analyze_query(customer.default_query(True, limit=args.limit))
            tsp.execute_query()
            records = tsp.get_batch()

            while records:
                record_counter += len(records)
                with TrainingSetGenerator(output_dir=output_dir, customer=customer) as g:
                    tfrecord = g.convert(records,
                                         batch_index=batch_counter,
                                         total_batch=tsp.number_of_batches,
                                         augment_cmd=augment)
                if tfrecord:
                    print(f"Successfully generated tfrecord: batch {batch_counter}, "
                          f"frames: {len(records)}, filename:{tfrecord}")
                    batch_counter += 1
                    records = tsp.get_batch()

    # print("Generating training set of {} frames into {} frames / batch tfrecord took"
    #       " {:f} seconds".format(num_records, int(args.batch_size), t.elapsed))

    # # Timer.log()
    # tfexp = DataExp(output_dir)
    # res = tfexp.analyze()
    # print(tfexp.num_records())
    # DataExp.plot_class_distribution(res, show_plot=True, save_dir=output_dir)
