from fedegs.federated.algorithms.fedavg import FedAvgServer
from fedegs.federated.algorithms.fedala import FedALAServer
from fedegs.federated.algorithms.confree import ConFREEServer
from fedegs.federated.algorithms.pfedfda import PFedFDAServer
from fedegs.federated.algorithms.fedegs import FedEGSServer
from fedegs.federated.algorithms.fedasym import FedAsymServer
from fedegs.federated.algorithms.fedasym_gain import FedAsymGainServer
from fedegs.federated.algorithms.fedegs2 import FedEGS2Server
from fedegs.federated.algorithms.fedegsbg import FedEGSBGServer
from fedegs.federated.algorithms.fedegsba import FedEGSBAServer
from fedegs.federated.algorithms.fedegsd import FedEGSDServer
from fedegs.federated.algorithms.fedegsd_s import FedEGSDSServer
from fedegs.federated.algorithms.fedegse import FedEGSEServer
from fedegs.federated.algorithms.fedegseu import FedEGSEUServer
from fedegs.federated.algorithms.fedegss import FedEGSSServer
from fedegs.federated.algorithms.fedegssl import FedEGSSLServer
from fedegs.federated.algorithms.fedegssg import FedEGSSGServer
from fedegs.federated.algorithms.fedprox import FedProxServer


def create_federated_server(name: str, config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer=None, public_dataset=None):
    normalized = name.lower()
    if normalized in {"fedegs", "fedavg_expert_slice", "fedprox_expert_slice", "fedegs_enhanced"}:
        return FedEGSServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized in {"fedegs2", "fedegs-2"}:
        return FedEGS2Server(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegsbg", "fedegs-bg", "fedegs_bg"}:
        return FedEGSBGServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedasym", "fed_asym", "fed-asym"}:
        return FedAsymServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedasym_gain", "fedasym-gain", "fedasymgain"}:
        return FedAsymGainServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegsd", "fedegs-d", "fedegs_d"}:
        return FedEGSDServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegsd-s", "fedegsd_s", "fedegsds"}:
        return FedEGSDSServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegss", "fedegs-s", "fedegs_s"}:
        return FedEGSSServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegse", "fedegs-e", "fedegs_e", "fedegsecho"}:
        return FedEGSEServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegseu", "fedegse-u", "fedegse_u", "fedegse-uar", "fedegse_uar"}:
        return FedEGSEUServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegssl", "fedegs-sl", "fedegs_sl"}:
        return FedEGSSLServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegssg", "fedegs-sg", "fedegs_sg"}:
        return FedEGSSGServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized in {"fedegsba", "fedegs-ba", "fedegs_ba"}:
        return FedEGSBAServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer, public_dataset=public_dataset)
    if normalized == "fedavg":
        return FedAvgServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized == "fedala":
        return FedALAServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized in {"confree", "con_free", "fedala_confree", "fedala-confree"}:
        return ConFREEServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized in {"pfedfda", "pfed-fda", "pfed_fda"}:
        return PFedFDAServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized in {"ideal", "ideal_upper_bound", "fat_client"}:
        return FedAvgServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    if normalized == "fedprox":
        return FedProxServer(config, client_datasets, client_test_datasets, data_module, test_hard_indices, writer)
    raise ValueError(f"Unsupported federated algorithm: {name}")
