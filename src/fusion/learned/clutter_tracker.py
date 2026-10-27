import numpy as np

from tracking.types import Detection, FrameResult
from tracking.models.motion import MotionModel
from fusion.models.radar import RadarRangeBearing
from tracking.tracker.track import Track
from tracking.association.assignment import hungarian_assign
from tracking.association.gating import chi2_gate_threshold, build_cost_matrix
from fusion.learned.clutter import build_cost_matrix_with_classifier, compute_features, init_track_from_detection

# MultiTargetTracker pasted below as placeholder
class MultiTargetTrackerClutter():
    """
    Multi-target tracker using a classifier-augmented association cost instead of pure kinematic gating.
    """

  
    def __init__(self, motion_model: MotionModel, radar_model: RadarRangeBearing, classifier=None, n_active_features: int = 0, n_confirm: int = 3, max_misses: int = 5, gate_confidence: float = .99, birth_threshold: float = 0.5):

        self.tracks: dict[int, Track] = {}
        self._next_id: int = 0
        self.motion = motion_model
        self.radar_model = radar_model
        self.classifier = classifier
        self.n_active_features = n_active_features
        self.n_confirm = n_confirm
        self.max_misses = max_misses
        self.gate_confidence = gate_confidence
        self.birth_threshold = birth_threshold

    def step(self, detections: list[Detection], dt: float, timestamp: float) -> FrameResult:
        """
        Actual step association step which utilizes cost matrix (with classifier), gating to keep plausbile pairs,
        and then uses Hungarian algorithm to solve resulting assingment

        In: detections, list[Detection]; this frame's raw, unlabeled detections
            dt; seconds since the last step
            timestamp; absolute current time (seconds)
        Out: FrameResult; snapshots of every live track (including those spawned within this frame),
                            n_detections, the track_ids (under assignments)
        """

        # For all tracks advance predict
        for track in self.tracks.values():
            track.predict(dt)

        # Build cost matrix
        track_ids_list = list(self.tracks.keys())
        tracks_list = list(self.tracks.values())
        gate = chi2_gate_threshold(self.radar_model.dim_z, self.gate_confidence)
        if self.classifier:
            cost = build_cost_matrix_with_classifier(tracks_list, detections, gate, self.classifier, self.n_active_features)
        else: 
            cost = build_cost_matrix(tracks_list, detections, gate)

        # Get matches, unmatched tracks, unmatched detections
        matches, unmatched_track_idx, unmatched_det_idx = hungarian_assign(cost)

        # Update matched tracks
        for row, col in matches:
            track_id = track_ids_list[row]
            self.tracks[track_id].update(detections[col])

        # Mark matches
        for row in unmatched_track_idx:
            track_id = track_ids_list[row]
            self.tracks[track_id].mark_missed()

        # Confirm if hits exceed confirmation
        for track in self.tracks.values():
            if track.hits >= self.n_confirm:
                track.confirmed = True

        # Create tentative tracks from unmatched detections
        for col in unmatched_det_idx:
            if self.classifier is not None:
                features = compute_features(detections[col], detections, tracks_list, gate)
                features = features[:self.n_active_features].reshape(1, -1)
                prob = self.classifier.predict_proba(features)[0, 1]
                if prob < self.birth_threshold:
                    continue  # classifier is confident this is clutter (no track spawn)
            new_id = self._next_id
            self._next_id += 1
            track = init_track_from_detection(detections[col], self.radar_model, self.motion, new_id, timestamp)
            self.tracks[new_id] = track

        # Delete dead tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].is_dead(self.max_misses):
                del self.tracks[track_id]


        # Build and return list of snapshots
        snapshots = [track.snapshot(timestamp) for track in self.tracks.values()]
        assignments = {track_ids_list[row]: col for row, col in matches}

        return FrameResult(timestamp=timestamp, snapshots=snapshots, n_detections=len(detections), assignments=assignments)

