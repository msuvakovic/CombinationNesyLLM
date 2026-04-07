"""
ROAD-R Data Loader
Parses road_trainval_v1.0.json into ASP facts for NeSyC continual learning.

ASP fact schema:
  agent(TubeId, AgentLabel, FrameId, VideoName).
  action(TubeId, ActionLabel, FrameId, VideoName).
  loc(TubeId, LocLabel, FrameId, VideoName).
  av_action(ActionLabel, FrameId, VideoName).   % ego-vehicle action

Continual learning phases (by label introduction):
  Phase 1 - Basic agents: Ped, Car, Cyc + actions: Mov, Stop, Brake
  Phase 2 - Extended agents: LarVeh, Bus, EmVeh + actions: Ovtak, IncatLeft, IncatRht
  Phase 3 - Full label set: all remaining agents, actions, locations
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional


# Labels introduced per phase (names must match dataset label strings)
PHASE_LABELS = {
    1: {
        "agents":  ["Ped", "Car", "Cyc"],
        "actions": ["Mov", "Stop", "Brake", "MovAway", "MovTow"],
        "locs":    ["VehLane", "Pav", "Jun"],
    },
    2: {
        "agents":  ["LarVeh", "Bus", "EmVeh", "Mobike", "MedVeh"],
        "actions": ["Ovtak", "IncatLft", "IncatRht", "Wait2X", "HazLit"],
        "locs":    ["OutgoLane", "IncomLane", "OutgoCycLane", "IncomCycLane"],
    },
    3: {
        "agents":  [],   # all remaining (TL, OthTL)
        "actions": ["TurLft", "TurRht", "Xing", "XingFmLft", "XingFmRht", "PushObj", "Red", "Amber", "Green"],
        "locs":    ["LftPav", "RhtPav", "xing", "BusStop", "parking"],
    },
}


# Explicit mapping from dataset label strings → ASP atoms
# Must match ROAD_LABELS in requirements.py exactly
_LABEL_ATOM_MAP = {
    # Agents
    "Ped": "ped", "Car": "car", "Cyc": "cyc", "Mobike": "mobike",
    "MedVeh": "med_veh", "LarVeh": "lar_veh", "Bus": "bus",
    "EmVeh": "em_veh", "TL": "tl", "OthTL": "oth_tl",
    # Actions
    "MovAway": "mov_away", "MovTow": "mov_tow", "Mov": "mov",
    "Brake": "brake", "Stop": "stop", "IncatLft": "incat_left",
    "IncatRht": "incat_rht", "Ovtak": "ovtak", "Wait2X": "wait2x",
    "TurLft": "tur_lft", "TurRht": "tur_rht",
    "XingFmLft": "xing_fm_lft", "XingFmRht": "xing_fm_rht",
    "Xing": "xing", "HazLit": "haz_lit", "Red": "red",
    "Amber": "amber", "Green": "green", "PushObj": "push_obj",
    "Rev": "rev", "MovRht": "mov_rht", "MovLft": "mov_lft",
    # Locations
    "VehLane": "veh_lane", "OutgoLane": "outgo_lane",
    "OutgoCycLane": "outgo_cyc_lane", "IncomLane": "incom_lane",
    "IncomCycLane": "incom_cyc_lane", "Pav": "pav",
    "LftPav": "lft_pav", "RhtPav": "rht_pav", "Jun": "jun",
    "xing": "xing_loc", "BusStop": "bus_stop", "parking": "parking",
}


def _sanitize(label: str) -> str:
    """Convert dataset label string to a valid ASP atom using explicit mapping.
    Falls back to lowercasing + replacing non-alphanum chars if not in map."""
    if label in _LABEL_ATOM_MAP:
        return _LABEL_ATOM_MAP[label]
    atom = re.sub(r'[^a-z0-9_]', '_', label.lower())
    # ASP atoms must not start with a digit or underscore
    if atom and (atom[0].isdigit() or atom[0] == '_'):
        atom = 't' + atom
    return atom


def _sanitize_id(uid: str) -> str:
    """Sanitize a tube/video ID for use as an ASP atom (must start with letter)."""
    atom = re.sub(r'[^a-z0-9_]', '_', str(uid).lower())
    if atom and (atom[0].isdigit() or atom[0] == '_'):
        atom = 'id_' + atom
    return atom


class ROADDataLoader:
    def __init__(self, annotation_path: str):
        """
        Args:
            annotation_path: Path to road_trainval_v1.0.json
        """
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}\n"
                "Download with: git clone https://github.com/gurkirt/road-dataset && "
                "bash road-dataset/get_dataset.sh"
            )
        with open(annotation_path, 'r') as f:
            self.data = json.load(f)

        self.agent_labels: List[str]  = self.data.get('agent_labels', [])
        self.action_labels: List[str] = self.data.get('action_labels', [])
        self.loc_labels: List[str]    = self.data.get('loc_labels', [])
        self.av_action_labels: List[str] = self.data.get('av_action_labels', [])
        self.all_labels: List[str]    = self.data.get('all_input_labels', [])
        self.database: Dict           = self.data.get('db', {})

        # Build cumulative label sets per phase
        self._phase_agent_sets  = self._build_cumulative(PHASE_LABELS, 'agents')
        self._phase_action_sets = self._build_cumulative(PHASE_LABELS, 'actions')
        self._phase_loc_sets    = self._build_cumulative(PHASE_LABELS, 'locs')

    def _build_cumulative(self, phase_labels: dict, key: str) -> Dict[int, set]:
        """Build cumulative label sets so phase N includes all labels from phases <= N."""
        cumulative = {}
        seen = set()
        for phase in sorted(phase_labels.keys()):
            seen.update(phase_labels[phase].get(key, []))
            cumulative[phase] = set(seen)
        # Phase 3 always includes everything
        all_labels = set(
            self.agent_labels + self.action_labels + self.loc_labels
        )
        cumulative[3] = all_labels
        return cumulative

    def get_video_names(self, split: str = 'train') -> List[str]:
        """
        Return video names for a split.
        ROAD-R split: 15 train, 3 val, 4 test (by convention, use first N videos).
        """
        all_videos = sorted(self.database.keys())
        if split == 'train':
            return all_videos[:15]
        elif split == 'val':
            return all_videos[15:18]
        elif split == 'test':
            return all_videos[18:]
        return all_videos

    def get_phase_videos(self, phase: int) -> List[str]:
        """Split training videos into three equal-ish groups for continual learning."""
        train_videos = self.get_video_names('train')
        chunk = len(train_videos) // 3
        if phase == 1:
            return train_videos[:chunk]
        elif phase == 2:
            return train_videos[chunk:2*chunk]
        else:
            return train_videos[2*chunk:]

    def annotations_to_asp_facts(
        self,
        video_name: str,
        phase: int = 3,
        max_frames: Optional[int] = None,
    ) -> str:
        """
        Convert all annotations in a video to ASP facts, filtered to labels
        available in the given phase.

        Returns a multi-line string of ASP facts.
        """
        if video_name not in self.database:
            raise KeyError(f"Video '{video_name}' not in annotation database.")

        allowed_agents  = self._phase_agent_sets.get(phase, set())
        allowed_actions = self._phase_action_sets.get(phase, set())
        allowed_locs    = self._phase_loc_sets.get(phase, set())

        vid_data = self.database[video_name]
        frames   = vid_data.get('frames', {})
        facts: List[str] = []

        # Video-level fact
        safe_vid = _sanitize_id(video_name)
        facts.append(f"video({safe_vid}).")

        frame_ids = sorted(frames.keys(), key=lambda x: int(x))
        if max_frames:
            frame_ids = frame_ids[:max_frames]

        for frame_id in frame_ids:
            frame = frames[frame_id]
            if not frame.get('annotated', 0):
                continue
            fid = int(frame_id)

            # AV (ego vehicle) action for this frame
            av_tubes = vid_data.get('av_action_tubes', {})
            for av_label_idx, tube_frames in av_tubes.items():
                if fid in tube_frames or str(fid) in tube_frames:
                    if int(av_label_idx) < len(self.av_action_labels):
                        av_label = self.av_action_labels[int(av_label_idx)]
                        facts.append(
                            f"av_action({_sanitize(av_label)}, {fid}, {safe_vid})."
                        )

            annos = frame.get('annos', {})
            for anno_key, anno in annos.items():
                tube_uid = _sanitize_id(str(anno.get('tube_uid', anno_key)))
                box = anno.get('box', [])

                # Bounding box fact — coords stored as integers (x10000) since
                # Clingo classical ASP does not support floating point literals.
                if len(box) == 4:
                    xmin, ymin, xmax, ymax = [int(round(v * 10000)) for v in box]
                    facts.append(
                        f"bbox({tube_uid}, {fid}, {safe_vid}, "
                        f"{xmin}, {ymin}, {xmax}, {ymax})."
                    )

                # Agent facts
                for aid in anno.get('agent_ids', []):
                    if aid < len(self.agent_labels):
                        label = self.agent_labels[aid]
                        if label in allowed_agents:
                            facts.append(
                                f"agent({tube_uid}, {_sanitize(label)}, {fid}, {safe_vid})."
                            )

                # Action facts
                for actid in anno.get('action_ids', []):
                    if actid < len(self.action_labels):
                        label = self.action_labels[actid]
                        if label in allowed_actions:
                            facts.append(
                                f"action({tube_uid}, {_sanitize(label)}, {fid}, {safe_vid})."
                            )

                # Location facts
                for lid in anno.get('loc_ids', anno.get('location_ids', [])):
                    if lid < len(self.loc_labels):
                        label = self.loc_labels[lid]
                        if label in allowed_locs:
                            facts.append(
                                f"loc({tube_uid}, {_sanitize(label)}, {fid}, {safe_vid})."
                            )

        return '\n'.join(facts)

    def get_phase_facts(self, phase: int, max_frames_per_video: Optional[int] = 100) -> str:
        """Return all ASP facts for the videos in a given phase."""
        videos = self.get_phase_videos(phase)
        all_facts = []
        for video in videos:
            try:
                facts = self.annotations_to_asp_facts(video, phase=phase, max_frames=max_frames_per_video)
                all_facts.append(f"% --- Video: {video} ---\n{facts}")
            except Exception as e:
                print(f"Warning: skipping video {video}: {e}")
        return '\n\n'.join(all_facts)

    def get_label_summary(self) -> Dict:
        """Return a summary of all labels in the dataset."""
        return {
            'n_agents':  len(self.agent_labels),
            'n_actions': len(self.action_labels),
            'n_locs':    len(self.loc_labels),
            'agents':    self.agent_labels,
            'actions':   self.action_labels,
            'locations': self.loc_labels,
            'n_videos':  len(self.database),
            'videos':    list(self.database.keys()),
        }

    def sample_trajectory(
        self,
        video_name: str,
        tube_uid: str,
        phase: int = 3,
        n_frames: int = 10,
    ) -> str:
        """
        Return a human-readable trajectory for a single agent tube across frames.
        Used for LLM prompting.
        """
        if video_name not in self.database:
            return ""
        frames = self.database[video_name].get('frames', {})
        trajectory = []
        frame_ids = sorted(frames.keys(), key=lambda x: int(x))

        allowed_agents  = self._phase_agent_sets.get(phase, set())
        allowed_actions = self._phase_action_sets.get(phase, set())
        allowed_locs    = self._phase_loc_sets.get(phase, set())

        count = 0
        for fid in frame_ids:
            if count >= n_frames:
                break
            frame = frames[fid]
            annos = frame.get('annos', {})
            for key, anno in annos.items():
                if _sanitize(str(anno.get('tube_uid', key))) != tube_uid:
                    continue
                agents  = [self.agent_labels[i] for i in anno.get('agent_ids', [])
                           if i < len(self.agent_labels) and self.agent_labels[i] in allowed_agents]
                actions = [self.action_labels[i] for i in anno.get('action_ids', [])
                           if i < len(self.action_labels) and self.action_labels[i] in allowed_actions]
                locs    = [self.loc_labels[i] for i in anno.get('loc_ids', anno.get('location_ids', []))
                           if i < len(self.loc_labels) and self.loc_labels[i] in allowed_locs]
                trajectory.append(
                    f"Frame {fid}: agent={agents}, actions={actions}, location={locs}"
                )
                count += 1
                break

        return '\n'.join(trajectory)
