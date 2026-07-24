from types import SimpleNamespace

import torch
import yaml

from lingbotvla.data.vla_data.utils import FeatureTransform


def _make_transform(tmp_path, robot_config, joints):
    robot_config["norm_stats"] = None
    config_path = tmp_path / "kuavo.yaml"
    config_path.write_text(yaml.safe_dump(robot_config), encoding="utf-8")
    data_config = SimpleNamespace(joints=joints, cameras=[], norm_type=[])
    return FeatureTransform(
        config_path,
        data_config,
        None,
        None,
        disabled_image_features=True,
        do_nomalize=False,
        chunk_size=2,
        return_item_befor_padding=True,
    )


def test_explicit_target_start_preserves_sparse_kuavo_slots(tmp_path):
    robot_config = {
        "states": [
            {
                "observation.state.arm.position": {
                    "origin_keys": [
                        {"observation.state": {"start": 0, "end": 7, "target_start": 7}}
                    ]
                }
            },
            {
                "observation.state.effector.position": {
                    "origin_keys": [
                        {"observation.state": {"start": 7, "end": 8, "target_start": 1}}
                    ]
                }
            },
        ],
        "actions": [
            {
                "action.arm.position": {
                    "origin_keys": [
                        {"action": {"start": 0, "end": 7, "target_start": 7}}
                    ],
                    "subtract_state": False,
                }
            },
            {
                "action.effector.position": {
                    "origin_keys": [
                        {"action": {"start": 7, "end": 8, "target_start": 1}}
                    ],
                    "subtract_state": False,
                }
            },
        ],
        "images": [],
    }
    transform = _make_transform(
        tmp_path,
        robot_config,
        ["{'arm.position': 14}", "{'effector.position': 2}"],
    )

    state = torch.arange(1, 9, dtype=torch.float32)
    action = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    converted = transform.convert_features(
        {"observation.state": state, "action": action},
        w_action=True,
    )

    assert converted["observation.state.arm.position"].shape == (14,)
    assert torch.equal(
        converted["observation.state.arm.position"],
        torch.cat([torch.zeros(7), state[:7]]),
    )
    assert torch.equal(
        converted["observation.state.effector.position"],
        torch.tensor([0.0, 8.0]),
    )
    assert converted["action.arm.position"].shape == (2, 14)
    assert torch.equal(
        converted["action.arm.position"],
        torch.cat([torch.zeros(2, 7), action[:, :7]], dim=-1),
    )
    assert torch.equal(
        converted["action.effector.position"],
        torch.stack([torch.zeros(2), action[:, 7]], dim=-1),
    )


def test_mapping_without_target_start_remains_compact(tmp_path):
    robot_config = {
        "states": [
            {
                "observation.state.arm.position": {
                    "origin_keys": [{"observation.state": {"start": 0, "end": 7}}]
                }
            }
        ],
        "actions": [
            {
                "action.arm.position": {
                    "origin_keys": [{"action": {"start": 0, "end": 7}}],
                    "subtract_state": False,
                }
            }
        ],
        "images": [],
    }
    transform = _make_transform(tmp_path, robot_config, ["{'arm.position': 14}"])
    state = torch.arange(7, dtype=torch.float32)
    action = torch.arange(14, dtype=torch.float32).reshape(2, 7)

    converted = transform.convert_features(
        {"observation.state": state, "action": action},
        w_action=True,
    )

    assert torch.equal(converted["observation.state.arm.position"], state)
    assert torch.equal(converted["action.arm.position"], action)


def test_multiple_origins_without_target_start_concatenate_bimanual_data(tmp_path):
    robot_config = {
        "states": [
            {
                "observation.state.arm.position": {
                    "origin_keys": [
                        {"observation.state": {"start": 0, "end": 7}},
                        {"observation.state": {"start": 8, "end": 15}},
                    ]
                }
            }
        ],
        "actions": [
            {
                "action.arm.position": {
                    "origin_keys": [
                        {"action": {"start": 0, "end": 7}},
                        {"action": {"start": 8, "end": 15}},
                    ],
                    "subtract_state": False,
                }
            }
        ],
        "images": [],
    }
    transform = _make_transform(tmp_path, robot_config, ["{'arm.position': 14}"])
    state = torch.arange(16, dtype=torch.float32)
    action = torch.arange(32, dtype=torch.float32).reshape(2, 16)

    converted = transform.convert_features(
        {"observation.state": state, "action": action},
        w_action=True,
    )

    assert torch.equal(
        converted["observation.state.arm.position"],
        torch.cat([state[:7], state[8:15]]),
    )
    assert torch.equal(
        converted["action.arm.position"],
        torch.cat([action[:, :7], action[:, 8:15]], dim=-1),
    )
