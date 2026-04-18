import hydra
import torch
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from roi_bytetrack import ROIByteTrack
from omegaconf import DictConfig, OmegaConf


def build_reid_model(cfg: DictConfig, device: torch.device):
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    if cfg.reid.type == "pretrained":
        base.fc = torch.nn.Identity()
        base.to(device).eval()
        return base

    if cfg.reid.type == "metric_learning":
        from metric import EmbeddingModel, TripletDataset, train_triplet

        roi_out = tuple(cfg.reid.roi_out)
        model = EmbeddingModel(
            base=base,
            num_classes=0,
            out_dim=cfg.reid.out_dim,
            roi_out=roi_out,
            roi_spatial_scale=cfg.reid.roi_spatial_scale,
        )

        if cfg.training.do_train:
            dataset = TripletDataset(dataset_dir=cfg.data.train_dir, max_frame_delta=cfg.training.max_k)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
            model.to(device)
            train_triplet(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                steps=cfg.training.steps,
                n_ids=cfg.training.n_ids,
                k_per_id=cfg.training.k_per_id,
                margin=cfg.training.margin,
                max_k=cfg.training.max_k,
                ce_weight=cfg.training.ce_weight,
                hard_negatives=cfg.training.hard_negatives,
                freeze_backbone=cfg.training.freeze_backbone,
                device=device,
            )
            torch.save(model.state_dict(), cfg.reid.checkpoint)
        else:
            model.load_state_dict(torch.load(cfg.reid.checkpoint, map_location=device))

        model.to(device).eval()
        return model

    raise ValueError(f"unknown reid.type: {cfg.reid.type!r}.")


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    det_model = YOLO(cfg.models["detection-model"])
    det_model.overrides["conf"] = cfg.detection.conf
    det_model.overrides["iou"] = cfg.detection.iou
    det_model.overrides["agnostic_nms"] = cfg.detection.agnostic_nms
    det_model.overrides["max_det"] = cfg.detection.max_det

    reid_model = build_reid_model(cfg, device)

    track = ROIByteTrack(model=det_model, reid_model=reid_model, device=device)

    track.process_tracking(
        cfg.data.dataset,
        cfg.output.images_tracked,
        cfg.output.metrics_path,
        use_roi=cfg.tracking.use_roi,
        roi_coef=cfg.tracking.roi_coef,
    )

    track.evaluate_mot(
        cfg.data.annotations,
        cfg.output.metrics_path,
        verbose=cfg.evaluation.verbose,
    )


if __name__ == "__main__":
    main()
