def false_track_rate(assignment_log: list[dict[int, int]], confirmed_track_ids_per_frame: list[set[int]]) -> float:
    """Fraction of confirmed tracks that were never matched to a real target.
    In:  assignment_log, list of per-frame {truth_id: track_id};
         confirmed_track_ids_per_frame, list of per-frame sets of confirmed track_ids
    Out: scalar in [0, 1], fraction of all track_ids ever confirmed during the run
         that never appeared as a matched value in assignment_log; NaN if no
         track was ever confirmed
    
    """

    # All track IDs ever matched to a real target
    matched_track_ids = set()
    for assignments in assignment_log:
        matched_track_ids.update(assignments.values())

    # All track IDs ever confirmed
    confirmed_track_ids = set()
    for track_ids in confirmed_track_ids_per_frame:
        confirmed_track_ids.update(track_ids)

    # Confirmed tracks that were never matched to a real target
    false_track_ids = confirmed_track_ids - matched_track_ids

    if len(confirmed_track_ids) == 0:
        return float('nan')

    return len(false_track_ids) / len(confirmed_track_ids)