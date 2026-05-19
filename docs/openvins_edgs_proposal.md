# OpenVINS + EDGS Proposal

This document preserves the original proposal that motivated the `frankenstein` pipeline work.

It was previously stored as `strat.md` at the repository root.

## Summary

The proposal is a proof-of-concept SLAM system that combines:

- OpenVINS as the real-time visual-inertial tracking backbone
- a Python Gaussian-splatting mapping backend
- an EDGS-inspired dense Gaussian initialization strategy
- ROS 2 as the integration layer for rapid prototyping

The original design goal was not to beat Photo-SLAM on runtime in the first prototype. The goal was to test whether a loosely coupled visual-inertial frontend plus dense, Python-driven Gaussian initialization is a viable and scientifically interesting direction for splatting-based SLAM.

Core hypothesis:

> Given reliable visual-inertial poses from OpenVINS, a denser and more structured Gaussian initialization strategy can produce better early-stage map quality, faster map usefulness, or higher rendering efficiency than conventional sparse or naive Gaussian initialization.

## Architecture

Frontend:

- OpenVINS
- IMU propagation
- visual feature tracking
- visual-inertial updates
- metric pose estimation

Middleware:

- ROS 2
- image, pose, calibration, and keyframe-packet transport
- explicit separation between tracking and mapping

Backend:

- Python mapping node
- dense correspondence matching
- triangulation
- Gaussian initialization
- asynchronous Gaussian refinement

## Original Research Position

The novelty is not only swapping ORB-SLAM for OpenVINS.

The intended novelty is a decoupled visual-inertial splatting SLAM system where dense Gaussian initialization, rather than tracker-coupled densification, is the primary research variable.

That makes the design meaningfully different from Photo-SLAM in terms of:

- stronger metric priors from VIO
- easier experimentation in Python
- explicit tracking/mapping separation
- direct study of Gaussian birth and initialization quality

## Assumptions

Technical assumptions:

- OpenVINS can provide sufficiently stable poses for downstream Gaussian mapping
- a ROS 2 prototype is acceptable even with overhead
- Python is acceptable for the first backend implementation
- EDGS-style initialization can be adapted from offline reconstruction into an incremental context

Research assumptions:

- better Gaussian initialization may matter more than tight coupling in early-stage map formation
- a slower but more modular prototype is acceptable for a viability study
- Photo-SLAM remains a fair baseline if evaluation conditions are controlled carefully

Evaluation assumptions:

- comparison should prioritize map quality, map formation speed, convergence behavior, splat efficiency, and tracking-conditioned performance over raw runtime alone

## Goals

Primary goal:

- demonstrate that an OpenVINS-driven, ROS2-connected Python Gaussian mapper with EDGS-inspired initialization is a viable splatting SLAM prototype

Secondary goals:

- show that dense initialization improves map quality over naive initialization
- show that visual-inertial tracking is sufficient as a strong pose prior
- build a modular research platform for later splatting SLAM experiments

Initial non-goals:

- matching Photo-SLAM runtime
- full loop closure integration
- a fully optimized end-to-end C++ implementation
- tight tracker-mapper feedback
- final publishable tuning

## Design Principle

Tracking must never wait for mapping.

OpenVINS remains the real-time state-estimation authority. The Gaussian mapper is asynchronous and may subsample, skip, or lag temporarily without blocking tracking.

## Relationship To Current Docs

The roadmap and current implementation status now live in:

- `openvins_edgs_roadmap.md`
- `openvins_replay_runbook.md`
- `phase2_packet_ingest_design.md`
- `phase3_edgs_packet_port.md`

This proposal is retained as background context rather than the active execution plan.
