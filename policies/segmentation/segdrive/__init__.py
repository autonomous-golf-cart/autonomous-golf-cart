"""segdrive: shared library for camera-only driving.

    cityscapes  class labels + colors
    segformer   load SegFormer and segment a frame
    bev         bird's-eye-view projection, overlay, road mask
    planning    lane-aware centerline / Frenet / MPC planners on the BEV
    steering    steering estimator + recorded control-log helpers
    draw        BEV drawing helpers (trajectories, routes, steering wheel)
    video       H.264 encoding via ffmpeg
    paths       calibration, recorded drive, and output locations

Submodules are imported explicitly (``from segdrive.bev import create_bev``)
so that light scripts don't pull in torch.
"""
