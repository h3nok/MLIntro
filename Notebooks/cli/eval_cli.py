import argparse
import os
import glob
from training_config import NeuralNetConfig
from customer.customers import CustomerObjectMap
from core.model.nn.neuralnet import NeuralNet as NN
from core.deployment.inference import InferenceModel as IM
from application.viNet import viNet
from common.io.interactive import query_tag, query_config
from core.database import database_interface as dbi
from common.db.stored_procedures import StoredProcedures as sp

parser = argparse.ArgumentParser(description='Deployment CLI')
parser.add_argument('--candidate_dir', default=r"E:\viNet_RnD\Research\Candidates\GWA\InceptionV3",
                    help="Absolute path to operationalized candidate model")
parser.add_argument('--customer', default='GWA', help="Customer of the network")
parser.add_argument('--tag', default=None, help="Evaluation tag")
parser.add_argument('--db_config', default=r'C:\ProgramData\viNet\config\database.ini',
                    help="Database config file to create connection string from")
parser.add_argument('--use_cpp', default=True, help="Formal verification. If set to False, "
                                                    "python will be used")

if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.exists(args.candidate_dir)

    model_pb = glob.glob(args.candidate_dir + "\\*.pb")[0]
    config = glob.glob(args.candidate_dir + "\\*.ini")[0]
    class_map = glob.glob(args.candidate_dir + "\\*.txt")[0]

    assert os.path.exists(model_pb)
    assert os.path.exists(class_map)
    assert os.path.exists(config)

    config = NeuralNetConfig(config)
    customer = CustomerObjectMap[args.customer]
    net = NN(config, customer)
    inference_model = IM(model_pb, class_map, net.input_shape[0])
    vinet = viNet(inference_model, customer().name)

    if args.use_cpp:
        server = dbi.PgsqlInterface()
        assert os.path.exists(args.db_config), f'{args.db_config} not found'
        server.connect(args.db_config)

        if not args.tag:
            args.tag = query_tag(server.connection, customer=customer)

        assert vinet.config in sp.vinet_configs(server.connection)

        print(f"Evaluating \'{vinet.config}\' on verification tag \'{args.tag}\'")
        vinet.verify(args.tag)
