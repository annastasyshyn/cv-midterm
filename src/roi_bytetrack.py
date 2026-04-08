import os
import cv2
import numpy as np
import pandas as pd
import motmetrics as mm
import supervision as sv
from ultralytics import YOLO
from ultralyticsplus import YOLO

import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import box_iou_batch
from supervision.tracker.byte_tracker import matching


class ROIByteTrack:
    def __init__(self,
                 model,
                 reid_model,
                 device = "cuda",):

        self.device = device
        self.reid_model = reid_model
        self.model = model
        
        self.reid_model.to(device)
        self.model.to(device)

        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )


    def extract_embeddings(self, frame, boxes):
        """
        frame: cv2 image
        boxes: numpy array (N, 4) в форматі xyxy
        """
        if len(boxes) == 0:
            return torch.empty((0, 2048)).to(self.device)

        rois = torch.zeros((len(boxes), 5)).to(self.device)
        rois[:, 1:] = torch.tensor(boxes).to(self.device)
        
        crops = []
        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                crop = np.zeros((256, 128, 3), dtype=np.uint8)

            crops.append(self.preprocess(crop))
        
        crops_tensor = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            embeddings = self.reid_model(crops_tensor)

        return embeddings

    def update_with_detections_roi(self, detections: Detections, frame = None, roi_coef = 0.5) -> Detections:
        """
        Took this from supervision library
        """
        tensors = np.hstack(
            (
                detections.xyxy,
                detections.confidence[:, np.newaxis],
            )
        )
        tracks = self.tracker.update_with_tensors(tensors=tensors)

        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

            if frame is not None:
                
                detection_embeddings = self.extract_embeddings(frame, detection_bounding_boxes)
                track_embeddings = self.extract_embeddings(frame, track_bounding_boxes)

                cos_sim = torch.cosine_similarity(
                    detection_embeddings.unsqueeze(1),  # (21, 1, D)
                    track_embeddings.unsqueeze(0),      # (1, 20, D)
                    dim=2
                ).cpu().numpy()

                ious = (1 - roi_coef) * ious + roi_coef * cos_sim

            iou_costs = 1 - ious

            matches, _, _ = matching.linear_assignment(iou_costs, 0.5)
            detections.tracker_id = np.full(len(detections), -1, dtype=int)
            for i_detection, i_track in matches:
                detections.tracker_id[i_detection] = int(
                    tracks[i_track].external_track_id
                )

            return detections[detections.tracker_id != -1]

        else:
            detections = Detections.empty()
            detections.tracker_id = np.array([], dtype=int)

            return detections


    def process_traching(self, 
                         source_dir, 
                         target_dir, 
                         mot_file_path = "detection_metrics.txt", 
                         use_roi = False, 
                         roi_coef = 0.5):

        image_paths = sorted([
            os.path.join(source_dir, f) for f in os.listdir(source_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open(mot_file_path, "w", encoding="utf-8") as mot_file:

            for frame_idx, img_path in enumerate(image_paths):
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                results = self.model(frame, verbose=False, imgsz=1024)[0] #train image size
                detections = sv.Detections.from_ultralytics(results)

                if use_roi:
                    detections = self.update_with_detections_roi(detections, frame, roi_coef)
                else:
                    detections = self.update_with_detections_roi(detections)

                for i in range(len(detections.xyxy)):
                    x1, y1, x2, y2 = detections.xyxy[i]

                    if detections.tracker_id is not None:
                        track_id = detections.tracker_id[i]
                        conf = detections.confidence[i]
                        w = x2 - x1
                        h = y2 - y1

                        line = f"{frame_idx+1},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                        mot_file.write(line)

                if detections.tracker_id is not None:
                    labels = [f"#{tid}" for tid in detections.tracker_id]
                    annotated_frame = self.box_annotator.annotate(frame.copy(), detections=detections)
                    annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
                    
                    output_path = os.path.join(target_dir, os.path.basename(img_path))
                    cv2.imwrite(output_path, annotated_frame)

                if frame_idx % 50 == 0:
                    print(f"Processed {frame_idx}/{len(image_paths)} images")
    
    @staticmethod
    def evaluate_mot(gt_file, res_file, verbose = True):
        gt_cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'trunc', 'occ']
        gt_df = pd.read_csv(gt_file, header=None, names=gt_cols)

        res_cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'v1', 'v2', 'v3']
        res_df = pd.read_csv(res_file, header=None, names=res_cols)

        acc = mm.MOTAccumulator(auto_id=True)
        frames = sorted(gt_df['frame'].unique())
        
        for frame_id in frames:
            gt_frame = gt_df[gt_df['frame'] == frame_id]
            res_frame = res_df[res_df['frame'] == frame_id]

            gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
            res_boxes = res_frame[['x', 'y', 'w', 'h']].values if not res_frame.empty else np.array([])

            distances = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)

            acc.update(
                gt_frame['id'].values,
                res_frame['id'].values if not res_frame.empty else [],
                distances
            )

        mh = mm.metrics.create()

        metrics_list = ['mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches']
        
        summary = mh.compute(acc, metrics=metrics_list, name='ByteTrack_Baseline')
        
        if verbose:
            str_summary = mm.io.render_summary(
                summary, 
                formatters=mh.formatters, 
                namemap=mm.io.motchallenge_metric_names
            )
            print("\n--- MOT Evaluation Results ---")
            print(str_summary)

        return summary

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO('mshamrai/yolov8n-visdrone') # model for detection
    model.overrides['conf'] = 0.25
    model.overrides['iou'] = 0.45
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000

    reid_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # model for feature extraction
    reid_model.fc = torch.nn.Identity()
    reid_model.to(device).eval()

    track = ROIByteTrack(model = model,
                         reid_model=reid_model,
                         device=device)
    
    mot_metricks_path = "detection_metrics.txt" # path where mot metricks will be stored, maybe will rework this
    
    track.process_traching("../task1/VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v", # path to dataset image sequence
                           "output_images_tracked",  #p ath to output tracked objects on the image
                           mot_metricks_path,
                           use_roi = True,
                           roi_coef = 0.3
                           )
    
    track.evaluate_mot("../task1/VisDrone2019-MOT-test-dev/annotations/uav0000009_03358_v.txt", #path to annotations
                       mot_metricks_path,
                       verbose = True
                       )

if __name__ == "__main__":
    main()
