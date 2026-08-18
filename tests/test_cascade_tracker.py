import os
import sys
from collections import deque
import numpy as np
import pytest

# Ensure src-python is in sys.path for direct editor/analyzer execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src-python'))

from magic.face import (  # type: ignore
    _box_center,
    _box_diagonal,
    _cascade_match_tracks,
    _cosine_similarity,
    _extract_embedding,
    _greedy_assignment,
    _optimal_assignment,
    _predict_track_position,
    _update_track_state,
    _match_tracks_to_detections,
    _build_tracks_from_seed_regions,
)


def _make_unit_vector(dim=512, seed=0):
    """Helper to create a deterministic unit-length feature embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class MockFace:
    """Mock TinyFace Face NamedTuple-like object."""

    def __init__(self, bbox=(10, 10, 80, 80), normed_emb=None):
        self.bounding_box = np.array(bbox, dtype=np.float32)
        self.bbox = self.bounding_box
        self.score = 0.99
        self.landmark_5 = np.zeros((5, 2), dtype=np.float32)
        self.normed_embedding = normed_emb
        self.embedding = normed_emb


def test_cosine_similarity():
    """Verify cosine similarity calculation."""
    v1 = _make_unit_vector(512, seed=1)
    v2 = _make_unit_vector(512, seed=2)

    # Identical vector -> 1.0
    assert abs(_cosine_similarity(v1, v1) - 1.0) < 1e-5

    # Orthogonal / different vectors
    sim = _cosine_similarity(v1, v2)
    assert -1.0 <= sim <= 1.0
    assert sim < 0.5  # Random high-dim unit vectors have near 0 similarity

    # None handling
    assert _cosine_similarity(v1, None) == 0.0
    assert _cosine_similarity(None, v2) == 0.0


def test_extract_embedding():
    """Verify safe extraction of normed_embedding."""
    emb = _make_unit_vector(512, seed=42)
    face = MockFace(normed_emb=emb)

    extracted = _extract_embedding(face)
    assert extracted is not None
    assert extracted.shape == (512,)
    assert np.allclose(extracted, emb)

    # Empty / invalid face
    assert _extract_embedding(None) is None
    assert _extract_embedding(object()) is None


def test_predict_track_position():
    """Verify velocity-based track position prediction."""
    # With no history, predicts center of current box
    track = {'box': (100, 100, 50, 50)}
    pred = _predict_track_position(track)
    assert pred == (125.0, 125.0)

    # With moving history: moving +10 x, +5 y per frame
    history = deque([(100.0, 100.0), (110.0, 105.0), (120.0, 110.0)], maxlen=5)
    track = {'box': (95, 85, 50, 50), 'box_history': history}
    pred = _predict_track_position(track)
    assert pred[0] > 120.0  # Extrapolated forward
    assert pred[1] > 110.0


def test_update_track_state():
    """Verify track state update on match."""
    emb1 = _make_unit_vector(512, seed=10)
    emb2 = _make_unit_vector(512, seed=11)

    track = {
        'trackId': 1,
        'box': (100, 100, 60, 60),
        'missed': 3,
        'seed_embedding': emb1,
        'ema_embedding': emb1.copy(),
        'box_history': deque([(130.0, 130.0)], maxlen=5),
    }

    new_box = (105, 102, 60, 60)
    _update_track_state(track, new_box, emb2)

    assert track['box'] == new_box
    assert track['missed'] == 0
    assert len(track['box_history']) == 2
    assert track['velocity'] == (5.0, 2.0)
    # EMA embedding should be blended
    assert track['ema_embedding'] is not None
    assert not np.allclose(track['ema_embedding'], emb1)
    assert not np.allclose(track['ema_embedding'], emb2)


def test_spatial_match_normal_movement():
    """Small movement between frames is matched via Stage 1 (spatial)."""
    emb_a = _make_unit_vector(512, seed=100)
    emb_b = _make_unit_vector(512, seed=200)

    tracks = {
        1: {
            'trackId': 1,
            'box': (100, 100, 80, 80),
            'missed': 0,
            'seed_embedding': emb_a,
            'ema_embedding': emb_a,
            'box_history': deque([(140.0, 140.0)], maxlen=5),
        },
        2: {
            'trackId': 2,
            'box': (400, 100, 80, 80),
            'missed': 0,
            'seed_embedding': emb_b,
            'ema_embedding': emb_b,
            'box_history': deque([(440.0, 140.0)], maxlen=5),
        },
    }

    # Frame t+1: slight movement (+5px, +2px)
    detections = [
        {'box': (105, 102, 80, 80), 'embedding': emb_a},
        {'box': (404, 103, 80, 80), 'embedding': emb_b},
    ]

    matches = _cascade_match_tracks(tracks, detections)
    match_dict = dict(matches)

    assert match_dict[1] == 0
    assert match_dict[2] == 1


def test_reid_recovers_after_large_displacement():
    """When a face undergoes a large displacement (e.g. shot cut or fast pan),
    Stage 1 spatial gate rejects it, but Stage 2 Feature Re-ID matches it correctly.
    """
    emb_person_a = _make_unit_vector(512, seed=101)
    emb_person_b = _make_unit_vector(512, seed=202)

    tracks = {
        1: {
            'trackId': 1,
            'box': (50, 50, 60, 60),  # Top-left
            'missed': 0,
            'seed_embedding': emb_person_a,
            'ema_embedding': emb_person_a,
            'box_history': deque([(80.0, 80.0)], maxlen=5),
        },
        2: {
            'trackId': 2,
            'box': (500, 50, 60, 60),  # Top-right
            'missed': 0,
            'seed_embedding': emb_person_b,
            'ema_embedding': emb_person_b,
            'box_history': deque([(530.0, 80.0)], maxlen=5),
        },
    }

    # Frame t+1: Person A and Person B swap positions or teleport to bottom of screen
    # Person A is now at (520, 400) - huge jump from (50, 50)
    # Person B is now at (80, 400) - huge jump from (500, 50)
    detections = [
        {'box': (80, 400, 60, 60), 'embedding': emb_person_b},   # det 0 is Person B
        {'box': (520, 400, 60, 60), 'embedding': emb_person_a},  # det 1 is Person A
    ]

    matches = _cascade_match_tracks(tracks, detections)
    match_dict = dict(matches)

    # Track 1 (Person A) must match det 1 (Person A)
    # Track 2 (Person B) must match det 0 (Person B)
    assert match_dict[1] == 1
    assert match_dict[2] == 0


def test_single_face_teleport_recovery():
    """When only 1 face appears in the frame after a huge position change,
    it should still be correctly re-identified and matched to the track.
    """
    emb_hero = _make_unit_vector(512, seed=777)

    tracks = {
        1: {
            'trackId': 1,
            'box': (50, 50, 80, 80),
            'missed': 2,
            'seed_embedding': emb_hero,
            'ema_embedding': emb_hero,
            'box_history': deque([(90.0, 90.0)], maxlen=5),
        },
    }

    # Only 1 detection across the entire screen at (700, 500)
    detections = [
        {'box': (700, 500, 80, 80), 'embedding': emb_hero},
    ]

    matches = _cascade_match_tracks(tracks, detections)
    assert len(matches) == 1
    assert matches[0] == (1, 0)


def test_multi_face_crossing_no_identity_switch():
    """When two people walk past each other (boxes overlap),
    feature similarity ensures their identities are not swapped.
    """
    emb_a = _make_unit_vector(512, seed=1)
    emb_b = _make_unit_vector(512, seed=2)

    # Person 1 moving right: (100 -> 150)
    # Person 2 moving left: (200 -> 150)
    tracks = {
        1: {
            'trackId': 1,
            'box': (120, 100, 80, 80),
            'missed': 0,
            'seed_embedding': emb_a,
            'ema_embedding': emb_a,
            'box_history': deque([(140.0, 140.0), (160.0, 140.0)], maxlen=5),
        },
        2: {
            'trackId': 2,
            'box': (180, 100, 80, 80),
            'missed': 0,
            'seed_embedding': emb_b,
            'ema_embedding': emb_b,
            'box_history': deque([(240.0, 140.0), (220.0, 140.0)], maxlen=5),
        },
    }

    # Both are now near x=150, but person 1 is at 170 and person 2 is at 130
    detections = [
        {'box': (130, 100, 80, 80), 'embedding': emb_b},  # det 0 is Person 2
        {'box': (170, 100, 80, 80), 'embedding': emb_a},  # det 1 is Person 1
    ]

    matches = _cascade_match_tracks(tracks, detections)
    match_dict = dict(matches)

    assert match_dict[1] == 1  # Track 1 -> Person 1
    assert match_dict[2] == 0  # Track 2 -> Person 2


def test_greedy_and_optimal_assignment():
    """Test assignment solvers on a synthetic cost matrix."""
    # Cost matrix: 2 tracks, 2 detections
    # Track 0 prefers det 1 (cost 0.1 vs 0.9)
    # Track 1 prefers det 0 (cost 0.2 vs 0.8)
    cost = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)

    greedy_res = _greedy_assignment(cost)
    optimal_res = _optimal_assignment(cost)

    assert set(greedy_res) == {(0, 1), (1, 0)}
    assert set(optimal_res) == {(0, 1), (1, 0)}


def test_backward_compatibility_legacy_tracks():
    """Tracks created without embedding or history dicts must still match without error."""
    legacy_tracks = {
        1: {'trackId': 1, 'box': (100, 100, 50, 50), 'missed': 0},
        2: {'trackId': 2, 'box': (300, 300, 50, 50), 'missed': 0},
    }

    detections = [
        {'box': (105, 105, 50, 50)},
        {'box': (305, 305, 50, 50)},
    ]

    matches = _match_tracks_to_detections(legacy_tracks, detections)
    match_dict = dict(matches)

    assert match_dict[1] == 0
    assert match_dict[2] == 1


def test_build_tracks_from_seed_regions():
    """Verify track initialization from seed regions creates all necessary tracking fields."""
    seed_regions = [
        {'x': 100, 'y': 100, 'width': 80, 'height': 80, 'faceSourceId': 'face_1'},
        {'x': 300, 'y': 100, 'width': 80, 'height': 80, 'faceSourceId': 'face_2'},
    ]

    emb = _make_unit_vector(512, seed=99)
    detections = [
        {'box': (102, 101, 80, 80), 'embedding': emb},
    ]

    tracks = _build_tracks_from_seed_regions(seed_regions, detections)
    assert len(tracks) == 2
    assert 1 in tracks
    assert 2 in tracks

    t1 = tracks[1]
    assert 'box_history' in t1
    assert 'velocity' in t1
    assert 'seed_embedding' in t1
    assert 'ema_embedding' in t1
    assert t1['seed_embedding'] is not None
