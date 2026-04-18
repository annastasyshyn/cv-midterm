import hydra
import torch
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights
from roi_bytetrack import ROIByteTrack
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="baseline")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO(cfg.models['detection-model'])
    model.overrides['conf'] = cfg.detection.conf
    model.overrides['iou'] = cfg.detection.iou
    model.overrides['agnostic_nms'] = cfg.detection.agnostic_nms
    model.overrides['max_det'] = cfg.detection.max_det

    reid_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    reid_model.fc = torch.nn.Identity()
    reid_model.to(device).eval()

    track = ROIByteTrack(model = model,
                         reid_model=reid_model,
                         device=device)
    
    mot_metricks_path = cfg.output.metrics_path
    
    track.process_tracking(cfg.data.dataset,
                           cfg.output.images_tracked,
                           mot_metricks_path,
                           use_roi = cfg.tracking.use_roi,
                           roi_coef = cfg.tracking.roi_coef
                           )

    track.evaluate_mot(cfg.data.annotations,
                       mot_metricks_path,
                       verbose = cfg.evaluation.verbose
                       )

if __name__ == "__main__":
    main()
