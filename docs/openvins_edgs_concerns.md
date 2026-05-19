# OpenVINS + EDGS Concerns

This note preserves the earlier concern log that compared the roadmap docs against the implementation through early Phase 3.

It was previously stored as `concerns.md` at the repository root.

## Resolution Summary

The original concern note identified three concrete mismatches. As of 2026-04-22:

- the Phase 1 `world_to_body` inverse-translation bug was fixed in the OpenVINS packet exporter path
- the Phase 2 frame-order bug was fixed in `gaussian-splatting/scene/dataset_readers.py`
- the Phase 1 wording was tightened so it describes the current export as replay-frame packet export, not as a frozen keyframe contract

The remaining gaps are mostly maturity issues:

- schema freezing
- validation
- Phase 3 still being only a partial packet-aware EDGS port

## Overall Assessment

The repo still broadly follows the intended roadmap through early Phase 3:

- Phase 0 baseline contracts are documented
- Phase 1 packet export exists
- Phase 2 packet-backed scene loading exists
- Phase 3 has started through packet-window restriction for EDGS

The earlier implementation/doc mismatches were real, but the two correctness issues that mattered most were fixed in code.

## Resolved Concerns

### Phase 1 `world_to_body` inverse correctness

Previous issue:

- docs claimed `world_to_body` was exported correctly
- code inverted rotation but reused the forward translation

Current state:

- fixed

Effect:

- the documented packet transform contract now matches exported data

### Phase 2 frame ordering semantics

Previous issue:

- docs said packet ordering should follow the packet stream
- loader re-sorted by `image_name`, which broke numeric frame order

Current state:

- fixed

Effect:

- loader now sorts by packet metadata:
  - `packet_index`
  - `camera_id`
  - `timestamp_sec` as a tiebreaker

### Phase 1 “keyframe packet” wording

Previous issue:

- docs implied the current export had a frozen keyframe policy
- implementation actually exports replay-time processed frames

Current state:

- fixed in documentation

Effect:

- the docs no longer overstate the maturity of the export contract

## Phase View

Phase 0:

- matches the current state well

Phase 1:

- now matches more closely after the transform fix and wording cleanup
- still lacks formal schema freezing and validation

Phase 2:

- now matches after the frame-order fix
- packet-backed training can run without COLMAP

Phase 3:

- started, but still only partially complete
- packet metadata exists on mapper cameras
- EDGS packet-window controls exist
- EDGS is not yet fully packet-native

## Bottom Line

The implementation remains aligned with the roadmap at a high level through Phase 3. The major correctness mismatches previously called out in this note are resolved.

The main remaining work is to mature the contract rather than to fix broken semantics.

## Recommended Follow-Up

1. Freeze the packet schema in a dedicated schema doc instead of leaving it implicit in code and phase notes.
2. Add packet validation for transforms, monotonic timestamps, and image presence.
3. Keep Phase 3 marked as partial until packet-native EDGS selection and ablations are actually complete.
