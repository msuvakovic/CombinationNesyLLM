"""
CLEVR-Hans Data Loader
Parses CLEVR-Hans3/7 scene JSON files into ASP facts for the SHIELD pipeline.

ASP fact schema per scene:
  scene(SceneId).
  obj(SceneId, ObjIdx, Shape, Size, Material, Color).
  left_of(SceneId, ObjA, ObjB).     % ObjA is left of ObjB
  right_of(SceneId, ObjA, ObjB).
  behind(SceneId, ObjA, ObjB).
  front_of(SceneId, ObjA, ObjB).
  class(SceneId, ClassId).           % ground truth label
  pixel_x(SceneId, ObjIdx, X).       % horizontal position (for left/right half)

Continual learning phases:
  Phase 1 — Class 0 only (large cube + large cylinder)
  Phase 2 — Classes 0+1 (add: small metal cube + small sphere)
  Phase 3 — Classes 0+1+2 (add: large blue sphere + small yellow sphere)
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

# ── Attribute value sets ─────────────────────────────────────────────────────
SHAPES    = {"cube", "sphere", "cylinder"}
SIZES     = {"large", "small"}
MATERIALS = {"metal", "rubber"}
COLORS    = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}

# Phase → classes introduced in that phase
PHASE_CLASSES = {1: [0], 2: [0, 1], 3: [0, 1, 2]}


def _sid(scene_id: int) -> str:
    return f"s{scene_id}"


def _oid(obj_idx: int) -> str:
    return f"o{obj_idx}"


class CLEVRHansLoader:
    def __init__(self, data_dir: str, variant: int = 3):
        """
        Args:
            data_dir: Path to the CLEVR-Hans3/ directory (contains train/, val/, test/)
            variant:  3 for CLEVR-Hans3, 7 for CLEVR-Hans7
        """
        self.data_dir = data_dir
        self.variant = variant
        self._scenes: Dict[str, List[dict]] = {}  # split → list of scenes

    def _scene_files(self, split: str) -> List[str]:
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        return sorted(
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.json')
        )

    def load_split(self, split: str = 'train', max_scenes: Optional[int] = None) -> List[dict]:
        """Load all scenes for a split. Caches result."""
        if split in self._scenes:
            return self._scenes[split][:max_scenes] if max_scenes else self._scenes[split]

        scenes = []
        for path in self._scene_files(split):
            with open(path) as f:
                data = json.load(f)
            # Each file may contain a list of scenes or a single scene dict
            if isinstance(data, list):
                scenes.extend(data)
            elif 'scenes' in data:
                scenes.extend(data['scenes'])
            else:
                scenes.append(data)

        self._scenes[split] = scenes
        return scenes[:max_scenes] if max_scenes else scenes

    def scene_to_asp(self, scene: dict, scene_id: int) -> str:
        """Convert a single scene dict to ASP facts."""
        sid = _sid(scene_id)
        facts = [f"scene({sid})."]

        # Ground truth class label
        if 'class_id' in scene:
            facts.append(f"class({sid}, c{scene['class_id']}).")

        objects = scene.get('objects', [])
        for i, obj in enumerate(objects):
            oid = _oid(i)
            shape    = obj.get('shape', 'unknown')
            size     = obj.get('size', 'unknown')
            material = obj.get('material', 'unknown')
            color    = obj.get('color', 'unknown')
            facts.append(f"obj({sid}, {oid}, {shape}, {size}, {material}, {color}).")

            # Pixel x-coordinate for left/right half reasoning (0-480 range)
            pixel_coords = obj.get('pixel_coords', [])
            if pixel_coords:
                px = int(pixel_coords[0])
                facts.append(f"pixel_x({sid}, {oid}, {px}).")
                # Derived: left/right half of 480px wide image
                if px < 240:
                    facts.append(f"in_left_half({sid}, {oid}).")
                else:
                    facts.append(f"in_right_half({sid}, {oid}).")

        # Spatial relationships
        rels = scene.get('relationships', {})
        rel_map = {
            'left':   'left_of',
            'right':  'right_of',
            'behind': 'behind',
            'front':  'front_of',
        }
        for rel_key, asp_pred in rel_map.items():
            rel_lists = rels.get(rel_key, [])
            for obj_a_idx, targets in enumerate(rel_lists):
                for obj_b_idx in targets:
                    facts.append(
                        f"{asp_pred}({sid}, {_oid(obj_a_idx)}, {_oid(obj_b_idx)})."
                    )

        return '\n'.join(facts)

    def split_to_asp(
        self,
        split: str = 'train',
        phase: int = 3,
        max_scenes: Optional[int] = None,
    ) -> str:
        """
        Convert all scenes in a split to ASP facts, filtered to classes
        available in the given phase.

        Args:
            split: 'train', 'val', or 'test'
            phase: 1, 2, or 3 — determines which classes are included
            max_scenes: limit number of scenes (for speed)
        """
        allowed_classes = PHASE_CLASSES.get(phase, [0, 1, 2])
        scenes = self.load_split(split, max_scenes=None)

        all_facts = []
        scene_counter = 0
        for scene in scenes:
            class_id = scene.get('class_id', -1)
            if class_id not in allowed_classes:
                continue
            all_facts.append(self.scene_to_asp(scene, scene_counter))
            scene_counter += 1
            if max_scenes and scene_counter >= max_scenes:
                break

        return '\n\n'.join(all_facts)

    def get_phase_split(
        self,
        phase: int,
        split: str = 'train',
        max_scenes: int = 200,
    ) -> str:
        """Return ASP facts for a given phase + split combination."""
        return self.split_to_asp(split=split, phase=phase, max_scenes=max_scenes)

    def get_sample_trajectory(self, phase: int, n_scenes: int = 5) -> str:
        """
        Return a human-readable trajectory of scenes for LLM prompting.
        """
        allowed_classes = PHASE_CLASSES.get(phase, [0, 1, 2])
        scenes = self.load_split('train')
        lines = [f"CLEVR-Hans scene examples (phase {phase}, classes {allowed_classes}):"]

        count = 0
        for scene in scenes:
            if scene.get('class_id', -1) not in allowed_classes:
                continue
            class_id = scene['class_id']
            objects = scene.get('objects', [])
            obj_descs = [
                f"{o.get('size','')} {o.get('color','')} {o.get('material','')} {o.get('shape','')}"
                for o in objects
            ]
            lines.append(f"  Class {class_id}: {' | '.join(obj_descs)}")
            count += 1
            if count >= n_scenes:
                break

        return '\n'.join(lines)

    def get_stats(self) -> dict:
        stats = {}
        for split in ['train', 'val', 'test']:
            try:
                scenes = self.load_split(split)
                from collections import Counter
                class_counts = Counter(s.get('class_id', -1) for s in scenes)
                stats[split] = {'total': len(scenes), 'by_class': dict(class_counts)}
            except FileNotFoundError:
                stats[split] = {'total': 0, 'by_class': {}}
        return stats


def generate_mock_clevr(output_dir: str, n_per_class: int = 100) -> str:
    """
    Generate a minimal mock CLEVR-Hans3 dataset for testing without downloading.
    Creates train/val/test splits with realistic scene structures.
    """
    import random
    random.seed(42)

    os.makedirs(output_dir, exist_ok=True)

    # Class 0: large cube + large cylinder (confound: cube is gray in train/val)
    # Class 1: small metal cube + small sphere (confound: sphere is metal in train/val)
    # Class 2: large blue sphere + small yellow sphere (no confound)
    def make_scene(class_id: int, split: str, idx: int) -> dict:
        objects = []
        rels = {'left': [], 'right': [], 'behind': [], 'front': []}

        if class_id == 0:
            color = 'gray' if split in ('train', 'val') else random.choice(list(COLORS))
            objects = [
                {'shape': 'cube', 'size': 'large', 'material': random.choice(['metal','rubber']),
                 'color': color, 'pixel_coords': [120, 240, 5.0], '3d_coords': [-1, 0, 0.5]},
                {'shape': 'cylinder', 'size': 'large', 'material': random.choice(['metal','rubber']),
                 'color': random.choice(list(COLORS - {'gray'})),
                 'pixel_coords': [360, 240, 5.0], '3d_coords': [1, 0, 0.5]},
            ]
        elif class_id == 1:
            mat = 'metal' if split in ('train', 'val') else random.choice(['metal', 'rubber'])
            objects = [
                {'shape': 'cube', 'size': 'small', 'material': 'metal',
                 'color': random.choice(list(COLORS)), 'pixel_coords': [150, 240, 4.0], '3d_coords': [-0.5, 0, 0.3]},
                {'shape': 'sphere', 'size': 'small', 'material': mat,
                 'color': random.choice(list(COLORS)), 'pixel_coords': [330, 240, 4.0], '3d_coords': [0.5, 0, 0.3]},
            ]
        else:  # class 2
            objects = [
                {'shape': 'sphere', 'size': 'large', 'material': random.choice(['metal','rubber']),
                 'color': 'blue', 'pixel_coords': [160, 240, 6.0], '3d_coords': [-1, 0, 0.7]},
                {'shape': 'sphere', 'size': 'small', 'material': random.choice(['metal','rubber']),
                 'color': 'yellow', 'pixel_coords': [320, 240, 3.5], '3d_coords': [1, 0, 0.3]},
            ]

        # Simple left/right relationships
        n = len(objects)
        for i in range(n):
            rels['left'].append([j for j in range(n) if j != i and objects[j]['pixel_coords'][0] < objects[i]['pixel_coords'][0]])
            rels['right'].append([j for j in range(n) if j != i and objects[j]['pixel_coords'][0] > objects[i]['pixel_coords'][0]])
            rels['behind'].append([])
            rels['front'].append([])

        return {
            'image_filename': f'CLEVR_HANS_{split}_{idx:06d}.png',
            'class_id': class_id,
            'objects': objects,
            'relationships': rels,
            'split': split,
        }

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        n = n_per_class if split == 'train' else n_per_class // 4

        for class_id in range(3):
            scenes = [make_scene(class_id, split, class_id * n + i) for i in range(n)]
            fname = os.path.join(split_dir, f'CLEVR_HANS_scenes_{split}_classid_{class_id}.json')
            with open(fname, 'w') as f:
                json.dump(scenes, f, indent=2)

    print(f"Mock CLEVR-Hans3 generated at {output_dir}")
    print(f"  {n_per_class * 3} train, {(n_per_class//4)*3} val, {(n_per_class//4)*3} test scenes")
    return output_dir
