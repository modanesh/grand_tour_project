"""Utility functions for OpenCV image processing."""

import cv2


def add_main_camera_text(img, step, cumulative_reward, actuation_mode, policy, exp_name=""):
    """Add overlay text to main scene camera image.

    Args:
        img: OpenCV image (BGR format)
        step: Current simulation step
        cumulative_reward: Cumulative reward value
        actuation_mode: Actuation mode string (e.g., "SEA", "PD", "Implicit")
        policy: Policy name string (e.g., "linear_regression", "ddpm")
        exp_name: Experiment name string (e.g., "dl_exp-1")
    """
    green = (0, 255, 0)  # BGR

    cv2.putText(
        img,
        f"step {step}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        green,
        2,
        cv2.LINE_AA,
    )
    if exp_name:
        cv2.putText(
            img,
            f"Exp: {exp_name}",
            (400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            green,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        f"cumulative reward: {round(cumulative_reward, 3)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        green,
        2,
        cv2.LINE_8,
    )
    cv2.putText(
        img,
        "Camera: Scene View",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        green,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Actuation: {actuation_mode}",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        green,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Policy: {policy}",
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        green,
        1,
        cv2.LINE_AA,
    )


def add_front_camera_text(img, step, actuation_mode, policy, exp_name=""):
    """Add overlay text to robot front camera image.

    Args:
        img: OpenCV image (BGR format)
        step: Current simulation step
        actuation_mode: Actuation mode string (e.g., "SEA", "PD", "Implicit")
        policy: Policy name string (e.g., "linear_regression", "ddpm")
        exp_name: Experiment name string (e.g., "dl_exp-1")
    """
    green = (0, 255, 0)  # BGR

    cv2.putText(
        img,
        f"step {step}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        green,
        2,
        cv2.LINE_AA,
    )
    if exp_name:
        cv2.putText(
            img,
            f"Exp: {exp_name}",
            (400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            green,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        "Camera: Robot Front View",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        green,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Actuation: {actuation_mode}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        green,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Policy: {policy}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        green,
        1,
        cv2.LINE_AA,
    )
