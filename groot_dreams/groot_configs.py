from groot_dreams.data.dataset import ModalityConfig
from groot_dreams.data.transform import (
    VideoCrop,
    VideoNormCrop,
    VideoResize,
    VideoThreePanelComposite,
    VideoToTensor,
)
from groot_dreams.data.transform.base import ComposedModalityTransform
from groot_dreams.data.transform.concat import ConcatTransform
from groot_dreams.data.transform.state_action import StateActionToTensor, StateActionTransform


def construct_modality_config_and_transforms(
    num_frames,
    embodiment,
    agibot_pad_freq10=False,
    waist_concat=False,
    camera_layout="head",
    head_crop_norm=None,
    wrist_crop_norm=None,
):
    if embodiment == "gr1":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.ego_view_freq20"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
        }
    elif embodiment == "g1":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.ego_view"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_leg",
                    "state.right_leg",
                    "state.waist",
                    "state.left_arm",
                    "state.left_hand",
                    "state.right_arm",
                    "state.right_hand",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_leg",
                    "action.right_leg",
                    "action.waist",
                    "action.left_arm",
                    "action.left_hand",
                    "action.right_arm",
                    "action.right_hand",
                ],
            ),
        }
    elif embodiment == "yam":
        timestep_interval = 4
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        if camera_layout == "three_panel":
            video_keys = [
                "video.top_camera-images-rgb",
                "video.left_camera-images-rgb",
                "video.right_camera-images-rgb",
            ]
        else:
            video_keys = ["video.top_camera-images-rgb"]
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.ee_pose_obs_left",
                    "state.ee_pose_obs_right",
                    "state.gripper_pos_obs_left",
                    "state.gripper_pos_obs_right",
                    "state.joint_pos_obs_left",
                    "state.joint_pos_obs_right",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.ee_pose_action_left",
                    "action.ee_pose_action_right",
                    "action.gripper_pos_action_left",
                    "action.gripper_pos_action_right",
                    "action.joint_pos_action_left",
                    "action.joint_pos_action_right",
                ],
            ),
        }
    elif embodiment == "agibot":
        timestep_interval = 4
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.top_head" if not agibot_pad_freq10 else "video.top_head_pad_freq10"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm_joint_position",
                    "state.right_arm_joint_position",
                    "state.left_effector_position",
                    "state.right_effector_position",
                    "state.head_position",
                    "state.waist_position",
                ] if waist_concat else [
                    "state.left_arm_joint_position",
                    "state.right_arm_joint_position",
                    "state.left_effector_position",
                    "state.right_effector_position",
                    "state.head_position",
                    "state.waist_pitch",
                    "state.waist_lift",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm_joint_position",
                    "action.right_arm_joint_position",
                    "action.left_effector_position",
                    "action.right_effector_position",
                    "action.head_position",
                    "action.waist_position",
                    "action.robot_velocity",
                ] if waist_concat else [
                    "action.left_arm_joint_position",
                    "action.right_arm_joint_position",
                    "action.left_effector_position",
                    "action.right_effector_position",
                    "action.head_position",
                    "action.waist_pitch",
                    "action.waist_lift",
                    "action.robot_velocity",
                ],
            ),
        }
    
    video_modality, state_modality, action_modality = config["video"], config["state"], config["action"]
    height = 480
    width = 640

    def _build_video_transforms_and_concat_order():
        if embodiment == "yam" and camera_layout == "head_crop":
            crop_box = tuple(head_crop_norm) if head_crop_norm is not None else (0.25, 0.32, 0.75, 0.92)
            v_transforms = [
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoNormCrop(apply_to=video_modality.modality_keys, crop_box=crop_box),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
            ]
            v_concat_order = video_modality.modality_keys
        elif embodiment == "yam" and camera_layout == "three_panel":
            h_crop = tuple(head_crop_norm) if head_crop_norm is not None else (0.1875, 0.35, 0.8125, 0.80)
            w_crop = tuple(wrist_crop_norm) if wrist_crop_norm is not None else (0.10, 278.0 / 1200.0, 0.90, 1.0)
            v_transforms = [
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoThreePanelComposite(
                    top_key="video.top_camera-images-rgb",
                    left_key="video.left_camera-images-rgb",
                    right_key="video.right_camera-images-rgb",
                    output_key="video.top_camera-images-rgb",
                    head_crop_norm=h_crop,
                    wrist_crop_norm=w_crop,
                    height=height,
                    width=width,
                    top_height=288,
                ),
            ]
            v_concat_order = ["video.top_camera-images-rgb"]
        else:
            v_transforms = [
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoCrop(apply_to=video_modality.modality_keys),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
            ]
            v_concat_order = video_modality.modality_keys
        return v_transforms, v_concat_order

    def _build_composed_transform():
        v_transforms, v_concat_order = _build_video_transforms_and_concat_order()
        return ComposedModalityTransform(
            transforms=[
                *v_transforms,
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(apply_to=state_modality.modality_keys, normalization_modes={
                    key: "min_max" for key in state_modality.modality_keys
                }),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                StateActionTransform(apply_to=action_modality.modality_keys, normalization_modes={
                    key: "min_max" for key in action_modality.modality_keys
                }),
                ConcatTransform(
                    video_concat_order=v_concat_order,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )

    train_transform = _build_composed_transform()
    test_transform = _build_composed_transform()

    return config, train_transform, test_transform
