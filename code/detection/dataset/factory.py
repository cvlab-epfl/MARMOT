from torch.utils.data import DataLoader, Subset

from detection.dataset.pomelo import PomeloSup, PomeloInf
from detection.dataset import heatmapbuilder
from detection.dataset.basedataset import FlowSceneSet
from detection.dataset.utils import get_train_val_split_index

from utils.log_utils import log, dict_to_string


def get_scene_set(data_arg, dataset_type):
    
    if dataset_type == "train":
        sceneset = PomeloSup(data_arg, "train")
    elif dataset_type == "val":
        sceneset = PomeloSup(data_arg, "val")
    elif dataset_type == "inference":
        sceneset = PomeloInf(data_arg)
    else:
        log.error(f"Unknown dataset type {dataset_type}")

    return sceneset    
 
def get_datasets(data_arg, dataset_type, use_aug, skip_init=0):

    sceneset = get_scene_set(data_arg, dataset_type)
    
    if data_arg["hm_type"] == "density":
        data_arg["hm_builder"] = heatmapbuilder.gaussian_density_heatmap
    elif data_arg["hm_type"] == "center":
        data_arg["hm_builder"] = heatmapbuilder.gausian_center_heatmap
    elif data_arg["hm_type"] == "constant":
        data_arg["hm_builder"] = heatmapbuilder.constant_center_heatmap
    else:
        log.error(f"Unknown heatmap type {data_arg['hm_type']}")

    dataset = FlowSceneSet(sceneset, data_arg, use_aug)

    collate_fn = dataset.collate_fn

    if skip_init != 0:
        dataset = Subset(dataset, range(skip_init, len(dataset)))
        
    return dataset, collate_fn


def get_dataloader(data_arg, skip_init=0):
    log.info(f"Building Datasets")
    log.debug(f"Data spec: {dict_to_string(data_arg)}")

    if data_arg["mode"] == "training":
        train_dataset, collate_fn_train = get_datasets(data_arg, "train", data_arg["aug_train"])
        train_dataset_no_aug, collate_fn_no_aug = get_datasets(data_arg, "train", False)
        val_dataset, collate_fn_val = get_datasets(data_arg, "val", False)
        
        if val_dataset.scene_set.get_ann_length(0) == 0:
            log.warning(f"Validation set contains no annotations")
            log.warning(f"Spliting training set as validation set, training split {data_arg['split_proportion']}")

            train_val_split = get_train_val_split_index(train_dataset, data_arg["split_proportion"])

            train_dataset = Subset(train_dataset, train_val_split[0])
            val_dataset = Subset(train_dataset_no_aug, train_val_split[1])

        train_dataloader = DataLoader(
                train_dataset,
                shuffle=data_arg["shuffle_train"],
                batch_size=data_arg["batch_size"],
                collate_fn=collate_fn_train,
                pin_memory=True,
                num_workers=data_arg["num_workers"]
                )
        
        val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=data_arg["batch_size"],
                collate_fn=collate_fn_no_aug,
                pin_memory=True,
                num_workers=data_arg["num_workers"]
                )

        return [train_dataloader], [val_dataloader]
    
    elif data_arg["mode"] == "evaluation":
        val_dataset, collate_fn = get_datasets(data_arg, "val", False, skip_init)

        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=data_arg["batch_size"],
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_arg["num_workers"]
            )

        return None, [val_dataloader]

    elif data_arg["mode"] == "inference":
        inf_dataset, collate_fn = get_datasets(data_arg, "inference", False, skip_init)

        inf_dataloader = DataLoader(
            inf_dataset,
            shuffle=False,
            batch_size=data_arg["batch_size"],
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_arg["num_workers"]
            )
        
        return None, [inf_dataloader]

    else:
        log.error(f"Unknown data_arg['mode']: {data_arg['mode']}")


