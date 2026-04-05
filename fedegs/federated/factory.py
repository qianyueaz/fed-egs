from fedegs.federated.algorithms.fedavg import FedAvgServer
from fedegs.federated.algorithms.fedegs import FedEGSServer
from fedegs.federated.algorithms.fedegs2 import FedEGS2Server
from fedegs.federated.algorithms.fedegs3 import FedEGS3Server
from fedegs.federated.algorithms.fedprox import FedProxServer


def create_federated_server(name: str, config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer=None, public_dataset=None):
    normalized = name.lower()
    if normalized in {"fedegs", "fedavg_expert_slice", "fedprox_expert_slice"}:
        return FedEGSServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized in {"fedegs2", "fedegs-2"}:
        return FedEGS2Server(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegs3", "fedegs-3"}:
        return FedEGS3Server(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized == "fedavg":
        return FedAvgServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized == "fedprox":
        return FedProxServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    raise ValueError(f"Unsupported federated algorithm: {name}")
