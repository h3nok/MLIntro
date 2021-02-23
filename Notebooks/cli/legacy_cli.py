from compat.legacy_model import LegacyModel
from customer.customers import CustomerObjectMap
import os
from common.io.interactive import query_yes_no

BASE_MODEL_DIR = r"E:\viNet_RnD\Deployment\Inference Models\Inception"


def create_candidate_model_legacy(class_map,
                                  network_version=None,
                                  dataset_version=None) -> None:
    """

    Used to freeze weights generated with legacy tool

    @param dataset_version:  dataset version
    @param network_version: network version
    @param class_map: a path to class map file

    @return: None
    """
    # select customer
    customers = list(CustomerObjectMap.keys())

    for i in range(len(customers)):
        print(f"{i}: {customers[i]}")
    print()

    customer = customers[int(input("Select customer (index):"))]
    freeze_latest = query_yes_no("Load latest checkpoints? ")

    if freeze_latest:
        # select base model
        models = os.listdir(BASE_MODEL_DIR)
        for i in range(len(models)):
            print(f"{i} - {models[i]}")

        input_graph_path = models[int(input("Enter net graph: "))]
        input_graph_path = os.path.join(BASE_MODEL_DIR, input_graph_path)
        assert os.path.exists(input_graph_path)

        model = LegacyModel(input_graph_path,
                            customer=customer,
                            net_version=network_version,
                            dataset_version=dataset_version)

        model.operationalize()
        model.upload(class_map)
        model.run_verification(customer=customer)
    else:
        model = LegacyModel("", customer=customer)
        model.run_verification()


if __name__ == '__main__':
    version, dsv = '2.11', 2
    classmap = r"E:\viNet_RnD\Deployment\Vattenfall\2.9\viNet_2.9_Vattenfall_ClassMap.txt"
    create_candidate_model_legacy(classmap,
                                  network_version=version,
                                  dataset_version=str(dsv))
