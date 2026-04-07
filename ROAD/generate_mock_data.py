"""
Generate a realistic mock road_trainval_v1.0.json for pipeline testing.

Uses the exact label names and structure from the real ROAD-R dataset
(derived from instance_counts.json and datasets.py in the ROAD-R repos).
Creates 5 mock videos with realistic annotation density.
"""

import json
import random
import os

random.seed(42)

# ── Exact label lists from the real ROAD-R dataset ──────────────────────────
AGENT_LABELS   = ["Ped", "Car", "Cyc", "Mobike", "MedVeh", "LarVeh", "Bus", "EmVeh", "TL", "OthTL"]
ACTION_LABELS  = ["MovAway", "MovTow", "Mov", "Brake", "Stop", "IncatLft", "IncatRht",
                  "Ovtak", "Wait2X", "TurLft", "TurRht", "XingFmLft", "XingFmRht",
                  "Xing", "HazLit", "Red", "Amber", "Green", "PushObj", "Rev"]
LOC_LABELS     = ["VehLane", "OutgoLane", "OutgoCycLane", "IncomLane", "IncomCycLane",
                  "Pav", "LftPav", "RhtPav", "Jun", "xing", "BusStop", "parking"]
AV_ACTION_LABELS = ["Mov", "Stop", "TurLft", "TurRht", "MovSlow"]

ALL_INPUT_LABELS = AGENT_LABELS + ACTION_LABELS + LOC_LABELS

# Realistic agent → (allowed_actions, allowed_locs) mappings
AGENT_PROFILES = {
    "Ped":    (["MovAway", "MovTow", "Stop", "Wait2X", "XingFmLft", "XingFmRht", "Xing", "PushObj"],
               ["Pav", "LftPav", "RhtPav", "xing", "BusStop"]),
    "Car":    (["MovAway", "MovTow", "Mov", "Brake", "Stop", "IncatLft", "IncatRht", "TurLft", "TurRht", "Ovtak"],
               ["VehLane", "OutgoLane", "IncomLane", "Jun"]),
    "Cyc":    (["MovAway", "MovTow", "Stop", "XingFmLft", "XingFmRht"],
               ["OutgoCycLane", "IncomCycLane", "LftPav", "RhtPav"]),
    "Mobike": (["MovAway", "MovTow", "Mov", "Stop", "TurLft"],
               ["VehLane", "OutgoLane"]),
    "MedVeh": (["MovAway", "MovTow", "Mov", "Brake", "Stop", "TurLft", "TurRht"],
               ["VehLane", "OutgoLane", "IncomLane", "Jun"]),
    "LarVeh": (["MovAway", "MovTow", "Stop", "Brake", "HazLit"],
               ["VehLane", "OutgoLane"]),
    "Bus":    (["MovAway", "MovTow", "Stop", "XingFmLft", "XingFmRht"],
               ["VehLane", "BusStop", "Jun"]),
    "EmVeh":  (["MovAway", "MovTow", "Mov", "HazLit"],
               ["VehLane", "OutgoLane", "IncomLane"]),
    "TL":     (["Red", "Amber", "Green"],
               ["Jun"]),
    "OthTL":  (["Red", "Amber", "Green"],
               ["Jun"]),
}

# ── Video name templates ─────────────────────────────────────────────────────
TRAIN_VIDEOS = [
    "2014-07-14-14-49-50_stereo_centre_01",
    "2014-08-08-13-15-11_stereo_centre_01",
    "2014-08-11-10-59-18_stereo_centre_02",
    "2014-11-14-16-34-33_stereo_centre_06",
    "2014-11-21-16-07-03_stereo_centre_01",
    "2014-12-10-18-10-50_stereo_centre_02",
    "2015-02-03-19-43-11_stereo_centre_04",
    "2015-02-06-13-57-16_stereo_centre_02",
    "2015-02-13-09-16-26_stereo_centre_05",
    "2015-02-24-12-32-19_stereo_centre_04",
    "2015-03-03-11-31-36_stereo_centre_01",
    "2015-03-03-11-31-36_stereo_centre_02",
    "2015-03-03-11-31-36_stereo_centre_03",
    "2015-04-24-17-26-06_stereo_centre_02",
    "2016-01-14-11-01-49_stereo_centre_01",
]
VAL_VIDEOS = [
    "2014-06-25-16-45-34_stereo_centre_02",
    "2014-07-14-14-49-50_stereo_centre_02",
    "2014-07-14-14-49-50_stereo_centre_03",
]
TEST_VIDEOS = [
    "2015-02-03-08-45-10_stereo_centre_02",
    "2015-02-06-13-57-16_stereo_centre_01",
    "2015-03-03-11-31-36_stereo_centre_04",
    "2016-01-14-11-01-49_stereo_centre_02",
]


def make_box():
    x1 = round(random.uniform(0.05, 0.6), 4)
    y1 = round(random.uniform(0.05, 0.6), 4)
    x2 = round(x1 + random.uniform(0.05, 0.35), 4)
    y2 = round(y1 + random.uniform(0.05, 0.35), 4)
    return [min(x1, 0.99), min(y1, 0.99), min(x2, 1.0), min(y2, 1.0)]


def make_annotation(agent_type, tube_uid, agent_labels, action_labels, loc_labels):
    profile_actions, profile_locs = AGENT_PROFILES.get(agent_type, (["Mov"], ["VehLane"]))

    # Pick 1 agent label
    agent_id = [agent_labels.index(agent_type)]

    # Pick 1-2 action labels from profile
    n_actions = random.randint(1, min(2, len(profile_actions)))
    picked_actions = random.sample(profile_actions, n_actions)
    action_ids = [action_labels.index(a) for a in picked_actions if a in action_labels]

    # Pick 1 location label from profile
    picked_loc = random.choice(profile_locs)
    loc_ids = [loc_labels.index(picked_loc)] if picked_loc in loc_labels else []

    return {
        "box": make_box(),
        "tube_uid": tube_uid,
        "agent_ids": agent_id,
        "action_ids": action_ids,
        "loc_ids": loc_ids,
        "duplex_ids": [],
        "triplet_ids": [],
    }


def make_video(video_name, n_frames=300, n_tubes=8, split_id=0):
    frames = {}
    agent_tubes = {}
    action_tubes = {}
    loc_tubes = {}
    av_action_tubes = {}

    # Create persistent tubes across frames
    tubes = []
    for i in range(n_tubes):
        agent = random.choice(AGENT_LABELS)
        uid = f"{video_name[:8]}_tube_{i:03d}"
        start = random.randint(0, n_frames // 4)
        length = random.randint(20, n_frames - start)
        tubes.append((uid, agent, start, min(start + length, n_frames)))

    for frame_num in range(1, n_frames + 1):
        # Annotate every 4th frame (sparse like real dataset)
        if frame_num % 4 != 0:
            continue

        annos = {}
        active_tubes = [(uid, ag, s, e) for uid, ag, s, e in tubes if s <= frame_num <= e]

        for uid, agent, _, _ in active_tubes:
            anno_key = f"{uid}_f{frame_num}"
            annos[anno_key] = make_annotation(
                agent, uid, AGENT_LABELS, ACTION_LABELS, LOC_LABELS
            )

        if annos:
            frames[str(frame_num)] = {
                "annotated": 1,
                "rgb_image_id": frame_num,
                "width": 1280,
                "height": 960,
                "annos": annos,
                "av_action_ids": [random.randint(0, len(AV_ACTION_LABELS) - 1)],
            }

    return {
        "numf": n_frames,
        "split_ids": [split_id],
        "frames": frames,
        "agent_tubes": {},
        "action_tubes": {},
        "loc_tubes": {},
        "duplex_tubes": {},
        "triplet_tubes": {},
        "av_action_tubes": {},
    }


def generate(output_path: str, frames_per_video: int = 200, tubes_per_video: int = 8):
    print(f"Generating mock road_trainval_v1.0.json → {output_path}")
    db = {}

    all_videos = (
        [(v, 0) for v in TRAIN_VIDEOS] +
        [(v, 1) for v in VAL_VIDEOS] +
        [(v, 2) for v in TEST_VIDEOS]
    )

    for i, (name, split_id) in enumerate(all_videos):
        print(f"  Generating video {i+1}/{len(all_videos)}: {name[:40]}...")
        db[name] = make_video(name, n_frames=frames_per_video,
                               n_tubes=tubes_per_video, split_id=split_id)

    annotation = {
        "label_types": ["agent", "action", "loc", "duplex", "triplet"],
        "all_input_labels": ALL_INPUT_LABELS,
        "all_agent_labels": AGENT_LABELS,
        "all_action_labels": ACTION_LABELS,
        "all_loc_labels": LOC_LABELS,
        "agent_labels": AGENT_LABELS,
        "action_labels": ACTION_LABELS,
        "loc_labels": LOC_LABELS,
        "av_action_labels": AV_ACTION_LABELS,
        "duplex_labels": [],
        "triplet_labels": [],
        "all_duplex_labels": [],
        "all_triplet_labels": [],
        "db": db,
    }

    with open(output_path, 'w') as f:
        json.dump(annotation, f)

    # Stats
    total_frames = sum(len(v["frames"]) for v in db.values())
    total_annos = sum(
        len(frame["annos"])
        for v in db.values()
        for frame in v["frames"].values()
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nDone!")
    print(f"  Videos:      {len(db)} ({len(TRAIN_VIDEOS)} train, {len(VAL_VIDEOS)} val, {len(TEST_VIDEOS)} test)")
    print(f"  Frames:      {total_frames}")
    print(f"  Annotations: {total_annos}")
    print(f"  File size:   {size_mb:.1f} MB")
    return output_path


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "data", "road_trainval_v1.0.json")
    generate(out, frames_per_video=300, tubes_per_video=10)
