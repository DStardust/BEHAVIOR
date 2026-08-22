"""Room adjacency derived from official room and traversability maps."""

from __future__ import annotations

from collections import deque

import numpy as np


def traversable_room_pairs(room_instance_map, traversability_map, room_name_to_id):
    """Return room pairs whose traversable regions meet along a valid route.

    Unlabelled traversable pixels at door thresholds are assigned by a
    multi-source flood fill.  When fronts from two rooms meet, those rooms are
    direct neighbours.  Walls remain blocked by the traversability map.
    """
    room_map = np.asarray(room_instance_map)
    free = np.asarray(traversability_map) != 0
    if room_map.ndim != 2 or free.ndim != 2:
        raise ValueError("room and traversability maps must be two-dimensional")
    if room_map.shape != free.shape:
        rows = np.minimum(
            (np.arange(free.shape[0]) * room_map.shape[0] / free.shape[0]).astype(int),
            room_map.shape[0] - 1,
        )
        cols = np.minimum(
            (np.arange(free.shape[1]) * room_map.shape[1] / free.shape[1]).astype(int),
            room_map.shape[1] - 1,
        )
        room_map = room_map[np.ix_(rows, cols)]

    id_to_name = {
        int(instance_id): str(name)
        for name, instance_id in room_name_to_id.items()
    }
    valid_ids = np.asarray(sorted(id_to_name), dtype=room_map.dtype)
    seeds = free & np.isin(room_map, valid_ids)
    owner = np.zeros(free.shape, dtype=np.int32)
    owner[seeds] = room_map[seeds].astype(np.int32)
    queue = deque(map(tuple, np.argwhere(seeds)))
    pairs = set()

    while queue:
        row, col = queue.popleft()
        source_id = int(owner[row, col])
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = row + drow, col + dcol
            if not (0 <= next_row < free.shape[0] and 0 <= next_col < free.shape[1]):
                continue
            if not free[next_row, next_col]:
                continue
            target_id = int(owner[next_row, next_col])
            if target_id == 0:
                owner[next_row, next_col] = source_id
                queue.append((next_row, next_col))
            elif target_id != source_id:
                left = id_to_name.get(source_id)
                right = id_to_name.get(target_id)
                if left and right:
                    pairs.add(tuple(sorted((left, right))))
    return pairs


def nearby_door_rooms(room_instance_map, center_pixel, radius_pixels, room_name_to_id, known_rooms=()):
    """Return up to two closest official room instances around a door."""
    room_map = np.asarray(room_instance_map)
    row, col = (int(center_pixel[0]), int(center_pixel[1]))
    radius = max(1, int(radius_pixels))
    row_start, row_stop = max(0, row - radius), min(room_map.shape[0], row + radius + 1)
    col_start, col_stop = max(0, col - radius), min(room_map.shape[1], col + radius + 1)
    patch = room_map[row_start:row_stop, col_start:col_stop]
    id_to_name = {int(instance_id): str(name) for name, instance_id in room_name_to_id.items()}
    ranked = []
    for instance_id in np.unique(patch):
        name = id_to_name.get(int(instance_id))
        if not name:
            continue
        offsets = np.argwhere(patch == instance_id)
        offsets[:, 0] += row_start
        offsets[:, 1] += col_start
        distance_sq = int(np.min(np.sum((offsets - np.asarray([row, col])) ** 2, axis=1)))
        ranked.append((distance_sq, name))

    selected = [name for name in known_rooms if name in room_name_to_id]
    for _, name in sorted(ranked):
        if name not in selected:
            selected.append(name)
        if len(selected) == 2:
            break
    return selected[:2]
