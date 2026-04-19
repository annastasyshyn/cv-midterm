import os
import cv2
import numpy as np
import pandas as pd
import motmetrics as mm
import supervision as sv
from ultralyticsplus import YOLO

import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from supervision.detection.core import Detections
from supervision.detection.utils.iou_and_nms import box_iou_batch
from supervision.tracker.byte_tracker.single_object_track import TrackState
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.core import joint_tracks, sub_tracks, remove_duplicate_tracks

from strack import STrack


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
    
    @staticmethod
    def embedding_distance(strack_pool, detections): #TODO: make not static

        strack_pool_len = len(strack_pool)
        detections_len = len(detections)

        if strack_pool_len == 0 or detections_len == 0:
            return np.zeros((strack_pool_len, detections_len), dtype=np.float32)

        #device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pool_feats = torch.stack([
            torch.as_tensor(t.curr_feat, dtype=torch.float32)#, device=device) 
            for t in strack_pool
        ])
        
        det_feats = torch.stack([
            torch.as_tensor(d.curr_feat, dtype=torch.float32)#, device=device) 
            for d in detections
        ])

        cos_sim = torch.cosine_similarity(
            pool_feats.unsqueeze(1),
            det_feats.unsqueeze(0),
            dim=2
        )

        cost_matrix = 1 - cos_sim
        
        cost_matrix = torch.clamp(cost_matrix, 0, 1)

        return cost_matrix.cpu().numpy()
        
    @staticmethod
    def update_with_tensors(self, tensors: np.ndarray, embeddings: np.ndarray = None, roi_coef = 0.5, ema_coef = 0.4) -> list[STrack]:

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = tensors[:, 4]
        bboxes = tensors[:, :4]

        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_keep,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                    curr_feat=embeddings[remain_inds][i] if embeddings is not None else None #my change is here
                )
                for i, (tlbr, score_keep) in enumerate(zip(dets, scores_keep))
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]

        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.shared_kalman)
        iou_dists = matching.iou_distance(strack_pool, detections)

        if embeddings is not None:
            emb_dists = ROIByteTrack.embedding_distance(strack_pool, detections) 
  
            dists = (1 - roi_coef) * iou_dists + roi_coef * emb_dists
        else:
            dists = iou_dists

        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, ema_coef)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_second,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                for (tlbr, score_second) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, ema_coef)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)

        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, ema_coef)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]

        return output_stracks

    def extract_embeddings(self, frame, boxes):
        """
        frame: cv2 image
        boxes: numpy array (N, 4) в форматі xyxy
        """
        if len(boxes) == 0:
            return torch.empty((0, 2048)).to(self.device)

        rois = torch.zeros((len(boxes), 5)).to(self.device)
        rois[:, 1:] = boxes
        
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

    def update_with_detections_roi(self, detections: Detections, embeddings = None, roi_coef = 0.5, ema_coef = 0.4) -> Detections:
        """
        Took this from supervision library
        """
        tensors = np.hstack(
            (
                detections.xyxy,
                detections.confidence[:, np.newaxis],
            )
        )
        tracks = ROIByteTrack.update_with_tensors(self.tracker,
                                                  tensors=tensors,
                                                  embeddings=embeddings,
                                                  roi_coef = roi_coef,
                                                  ema_coef = ema_coef)

        if len(tracks) > 0:
            detection_bounding_boxes = np.asarray([det[:4] for det in tensors])
            track_bounding_boxes = np.asarray([track.tlbr for track in tracks])

            ious = box_iou_batch(detection_bounding_boxes, track_bounding_boxes)

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
                         roi_coef = 0.5,
                         ema_coef = 0.4):

        image_paths = sorted([
            os.path.join(source_dir, f) for f in os.listdir(source_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        with open(mot_file_path, "w", encoding="utf-8") as mot_file:

            for frame_idx, img_path in enumerate(image_paths):
                frame = cv2.imread(img_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame is None:
                    continue

                results = self.model(frame, verbose=False, imgsz=1024)[0] #train image size
                detections = sv.Detections.from_ultralytics(results)

                if use_roi:
                    boxes = results.boxes.data[:, :4]
                    
                    embeddings = self.extract_embeddings(frame, boxes)
                    
                    detections = self.update_with_detections_roi(detections, embeddings, roi_coef, ema_coef=ema_coef)
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
                         device=device
                         )
    
    mot_metricks_path = "detection_metrics.txt" # path where mot metricks will be stored, maybe will rework this
    
    roi_coef = 0.25
    ema_coef = 0.4

    track.process_traching("../task1_2/VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v", # path to dataset image sequence
                           "output_images_tracked",  #path to output tracked objects on the image
                           mot_metricks_path,
                           use_roi = True,
                           roi_coef = roi_coef,
                           ema_coef = ema_coef
                           )

    track.evaluate_mot("../task1_2/VisDrone2019-MOT-test-dev/annotations/uav0000009_03358_v.txt", #path to annotations
                       mot_metricks_path,
                       verbose = True
                       )

if __name__ == "__main__":
    main()
