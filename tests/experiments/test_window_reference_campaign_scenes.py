from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.window_reference_campaign.config import (
    CampaignOverrides,
    DatasetKind,
    EvaluationKind,
    PresetSceneConfig,
    load_campaign_config,
)
from experiments.window_reference_campaign.scenes import (
    ResolvedFrameSelection,
    ResolvedScene,
    apply_frame_selection,
    resolve_kitti_layout,
    resolve_scene,
)
from experiments.window_reference_campaign.staging import (
    _validate_raw_dataset_output,
    guarded_remove,
    prepare_pointcloud_gt,
    require_descendant,
    stage_scene,
)
import experiments.window_reference_campaign.staging as staging_module


def _png(path: Path, colour: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color=(colour, colour, colour)).save(path)
    return path


@pytest.mark.parametrize("official", [False, True])
def test_kitti_resolver_accepts_normalized_and_official_layout(tmp_path, official):
    if official:
        image_dir = tmp_path / "dataset/sequences/04/image_2"
        poses = tmp_path / "dataset/poses/04.txt"
    else:
        image_dir = tmp_path / "04/image_2"
        poses = tmp_path / "04/poses.txt"
    _png(image_dir / "10.png")
    _png(image_dir / "2.png")
    poses.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(poses, np.tile(np.arange(12, dtype=float), (2, 1)))

    layout = resolve_kitti_layout(tmp_path, "04")

    assert layout.image_dir == image_dir.resolve()
    assert layout.poses_path == poses.resolve()
    assert layout.layout_name == ("official" if official else "normalized")


def test_frame_selection_is_applied_once_to_the_explicit_source_vector():
    selected = apply_frame_selection(
        tuple(range(0, 100, 10)),
        PresetSceneConfig(scene_id="fixture", start=1, stop=9, stride=2),
        start_override=None,
        max_frames=3,
        stride_override=None,
    )
    assert selected == ResolvedFrameSelection(
        start=1,
        stop=7,
        stride=2,
        source_frame_ids=(10, 30, 50),
    )


def test_resolve_scene_naturally_orders_kitti_and_keeps_dense_source_ids(tmp_path):
    root = tmp_path / "KITTI/04/image_2"
    _png(root / "10.png", 10)
    _png(root / "2.png", 2)
    _png(root / "1.png", 1)
    poses = root.parent / "poses.txt"
    poses.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(poses, np.arange(36, dtype=float).reshape(3, 12))
    loaded = load_campaign_config(
        Path(__file__).resolve().parents[2]
        / "configs/experiments/window_reference_campaign.yaml",
        CampaignOverrides(preset="kitti-smoke", data_root=tmp_path),
    )

    resolved = resolve_scene(loaded.config, loaded.config.selected_scenes[0])

    assert resolved.selection.source_frame_ids == (0, 1, 2)
    assert [path.name for path in resolved.source_images] == ["1.png", "2.png", "10.png"]
    assert resolved.evaluation_kind is EvaluationKind.INTERNAL_TRAJECTORY
    assert resolved.poses_path == poses.resolve()


def test_resolve_scene_reads_only_named_sequence_map_entry(tmp_path):
    root = tmp_path / "NeuralRGBD/thin_geometry"
    for frame_id in (0, 10, 20):
        _png(root / "images" / f"img{frame_id}.png", frame_id)
        _png(root / "depth" / f"depth{frame_id}.png", frame_id)
    np.savetxt(root / "poses.txt", np.tile(np.eye(4), (21, 1)))
    sequence_map = tmp_path / "map.json"
    sequence_map.write_text(
        json.dumps({"thin_geometry": list(range(0, 400, 10)), "broken": {"not": "a vector"}}),
        encoding="utf-8",
    )
    config_path = Path(__file__).resolve().parents[2] / "configs/experiments/window_reference_campaign.yaml"
    loaded = load_campaign_config(config_path, CampaignOverrides(preset="pointcloud-small", data_root=tmp_path))
    scene_config = replace(
        loaded.config.scenes["nrgbd-thin-geometry"],
        frame_index_map=str(sequence_map),
    )
    config = replace(
        loaded.config,
        scenes={**loaded.config.scenes, "nrgbd-thin-geometry": scene_config},
        selected_scenes=(PresetSceneConfig("nrgbd-thin-geometry", 0, 3, 1),),
    )

    resolved = resolve_scene(config, config.selected_scenes[0])

    assert resolved.selection.source_frame_ids == (0, 10, 20)
    assert resolved.expected_gt_shape == (392, 518)
    assert resolved.evaluation_kind is EvaluationKind.POINTCLOUD


def _kitti_scene(tmp_path: Path) -> ResolvedScene:
    root = tmp_path / "data"
    images = tuple(
        _png(root / "04/image_2" / f"{index:06d}.png", index)
        for index in range(6)
    )
    np.savetxt(root / "04/poses.txt", np.arange(6 * 12, dtype=float).reshape(6, 12))
    return ResolvedScene(
        scene_id="kitti-04",
        dataset=DatasetKind.KITTI,
        scene="04",
        slice_id="f000001-000006-s2",
        approved_data_root=root.resolve(),
        source_images=tuple(images[index] for index in (1, 3, 5)),
        selection=ResolvedFrameSelection(1, 6, 2, (1, 3, 5)),
        evaluation_kind=EvaluationKind.INTERNAL_TRAJECTORY,
        poses_path=(root / "04/poses.txt").resolve(),
        prepared_gt_path=None,
        frame_index_map=None,
        expected_gt_shape=None,
    )


def test_staging_uses_one_vector_for_images_pose_rows_and_manifest(tmp_path):
    staged = stage_scene(_kitti_scene(tmp_path), tmp_path / "campaign")

    assert sorted(path.name for path in staged.image_dir.iterdir()) == [
        "000000.png", "000001.png", "000002.png"
    ]
    assert all(path.is_symlink() for path in staged.image_dir.iterdir())
    np.testing.assert_array_equal(
        np.load(Path(staged.manifest_path).parent / "source_frame_ids.npy"),
        [1, 3, 5],
    )
    poses = np.atleast_2d(np.loadtxt(staged.poses_path))
    np.testing.assert_array_equal(
        poses,
        np.arange(6 * 12, dtype=float).reshape(6, 12)[[1, 3, 5]],
    )
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_frame_ids"] == [1, 3, 5]
    assert manifest["pipeline_sample_stride"] == 1


def test_staging_reuses_only_matching_hashes(tmp_path):
    scene = _kitti_scene(tmp_path)
    first = stage_scene(scene, tmp_path / "campaign")
    second = stage_scene(scene, tmp_path / "campaign")
    assert second.manifest_sha256 == first.manifest_sha256
    (second.image_dir / "000001.png").unlink()
    with pytest.raises(ValueError, match="staging file hash mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def test_staging_rejects_symlink_target_outside_approved_data_root(tmp_path):
    scene = _kitti_scene(tmp_path)
    escaped = tmp_path / "outside.png"
    _png(escaped)
    scene = replace(scene, source_images=(escaped, *scene.source_images[1:]))
    with pytest.raises(ValueError, match="approved data root"):
        stage_scene(scene, tmp_path / "campaign")


def _pointcloud_scene(
    tmp_path: Path,
    *,
    ids: tuple[int, ...],
    requested: tuple[int, ...],
) -> ResolvedScene:
    root = tmp_path / "data"
    images = tuple(
        _png(root / "NeuralRGBD/thin_geometry/images" / f"img{frame_id}.png", frame_id)
        for frame_id in ids
    )
    prepared = root / "prepared/ground_truth.npz"
    return ResolvedScene(
        scene_id="nrgbd-thin-geometry",
        dataset=DatasetKind.NRGBD,
        scene="thin_geometry",
        slice_id="f000001-000003-s1",
        approved_data_root=root.resolve(),
        source_images=tuple(images[ids.index(frame_id)] for frame_id in requested),
        selection=ResolvedFrameSelection(1, 3, 1, requested),
        evaluation_kind=EvaluationKind.POINTCLOUD,
        poses_path=None,
        prepared_gt_path=prepared,
        frame_index_map=None,
        expected_gt_shape=(392, 518),
    )


def test_external_pointcloud_gt_is_selected_by_exact_frame_ids(tmp_path):
    scene = _pointcloud_scene(tmp_path, ids=(0, 10, 20), requested=(10, 20))
    prepared = scene.prepared_gt_path
    assert prepared is not None
    prepared.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prepared,
        point_maps=np.stack([
            np.full((392, 518, 3), value, dtype=np.float32)
            for value in (0.0, 10.0, 20.0)
        ]),
        valid_mask=np.ones((3, 392, 518), dtype=bool),
        frame_ids=np.array([0, 10, 20], dtype=np.int64),
    )

    staged = stage_scene(scene, tmp_path / "campaign")
    with np.load(staged.pointcloud_gt_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_ids"], [10, 20])
        assert data["point_maps"].shape == (2, 392, 518, 3)
        assert np.all(data["point_maps"][0] == 10.0)


def test_pointcloud_gt_rejects_leading_dimension_or_spatial_mismatch(tmp_path):
    scene = _pointcloud_scene(tmp_path, ids=(0, 10), requested=(0, 10))
    write_external_gt = scene.prepared_gt_path
    assert write_external_gt is not None
    write_external_gt.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        write_external_gt,
        point_maps=np.zeros((1, 391, 518, 3), dtype=np.float32),
        valid_mask=np.ones((1, 391, 518), dtype=bool),
        frame_ids=np.array([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="point-map GT.*frames|spatial shape"):
        stage_scene(scene, tmp_path / "campaign")


def test_pointcloud_gt_accepts_sibling_frame_ids_and_writes_atomic_payload(tmp_path):
    scene = _pointcloud_scene(tmp_path, ids=(0, 10), requested=(10,))
    prepared = scene.prepared_gt_path
    assert prepared is not None
    prepared.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prepared,
        point_maps=np.ones((2, 392, 518, 3), dtype=np.float32),
        valid_mask=np.ones((2, 392, 518), dtype=bool),
    )
    np.save(prepared.parent / "source_frame_ids.npy", np.array([0, 10], dtype=np.int64))

    output, shape = prepare_pointcloud_gt(scene, tmp_path / "campaign/gt.npz")

    assert output == tmp_path / "campaign/gt.npz"
    assert shape == (1, 392, 518, 3)
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_ids"], [10])


def test_pointcloud_gt_rejects_disagreeing_frame_id_sources(tmp_path):
    scene = _pointcloud_scene(tmp_path, ids=(0, 10), requested=(10,))
    prepared = scene.prepared_gt_path
    assert prepared is not None
    prepared.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prepared,
        point_maps=np.ones((2, 392, 518, 3), dtype=np.float32),
        valid_mask=np.ones((2, 392, 518), dtype=bool),
        frame_ids=np.array([0, 10], dtype=np.int64),
    )
    np.save(prepared.parent / "source_frame_ids.npy", np.array([0, 11], dtype=np.int64))

    with pytest.raises(ValueError, match="frame ID sources disagree"):
        stage_scene(scene, tmp_path / "campaign")


def test_staging_requires_exact_generated_file_manifest_and_physical_set(tmp_path):
    scene = _kitti_scene(tmp_path)
    staged = stage_scene(scene, tmp_path / "campaign")
    manifest_path = Path(staged.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("poses.txt")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="staging file hash mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def test_staging_rejects_required_file_deleted_even_when_manifest_records_it(tmp_path):
    scene = _kitti_scene(tmp_path)
    staged = stage_scene(scene, tmp_path / "campaign")
    manifest_path = Path(staged.manifest_path)
    Path(staged.poses_path).unlink()

    with pytest.raises(ValueError, match="staging file hash mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def test_staging_rejects_unexpected_generated_file_entry(tmp_path):
    scene = _kitti_scene(tmp_path)
    staged = stage_scene(scene, tmp_path / "campaign")
    manifest_path = Path(staged.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra = manifest_path.parent / "unexpected.bin"
    extra.write_bytes(b"unexpected")
    manifest["files"]["unexpected.bin"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="staging file hash mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def test_external_sibling_frame_id_mutation_invalidates_existing_staging(tmp_path):
    scene = _pointcloud_scene(tmp_path, ids=(0, 10, 20), requested=(10, 20))
    prepared = scene.prepared_gt_path
    assert prepared is not None
    prepared.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prepared,
        point_maps=np.ones((3, 392, 518, 3), dtype=np.float32),
        valid_mask=np.ones((3, 392, 518), dtype=bool),
    )
    sibling = prepared.parent / "source_frame_ids.npy"
    np.save(sibling, np.array([0, 10, 20], dtype=np.int64))
    stage_scene(scene, tmp_path / "campaign")
    np.save(sibling, np.array([0, 11, 20], dtype=np.int64))

    with pytest.raises(ValueError, match="staging manifest mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def _raw_nrgbd_scene(tmp_path: Path) -> ResolvedScene:
    root = tmp_path / "raw-data"
    scene_dir = root / "NeuralRGBD/selected"
    image = _png(scene_dir / "images/img0.png", 2)
    _png(scene_dir / "depth/depth0.png", 1)
    # Replace the RGB fixture with an actual 16-bit depth image.
    from PIL import Image

    Image.fromarray(np.full((480, 640), 1000, dtype=np.uint16)).save(
        scene_dir / "depth/depth0.png"
    )
    np.savetxt(scene_dir / "poses.txt", np.eye(4))
    return ResolvedScene(
        scene_id="raw-nrgbd",
        dataset=DatasetKind.NRGBD,
        scene="selected",
        slice_id="f000000-000001-s1",
        approved_data_root=root.resolve(),
        source_images=(image,),
        selection=ResolvedFrameSelection(0, 1, 1, (0,)),
        evaluation_kind=EvaluationKind.POINTCLOUD,
        poses_path=None,
        prepared_gt_path=None,
        frame_index_map=None,
        expected_gt_shape=(392, 518),
    )


@pytest.mark.parametrize("dependency", ["depth", "poses"])
def test_raw_nrgbd_dependency_mutation_invalidates_existing_staging(tmp_path, dependency):
    scene = _raw_nrgbd_scene(tmp_path)
    stage_scene(scene, tmp_path / "campaign")
    if dependency == "depth":
        _png(scene.source_images[0].parent.parent / "depth/depth0.png", 9)
        from PIL import Image

        Image.fromarray(np.full((480, 640), 2000, dtype=np.uint16)).save(
            scene.source_images[0].parent.parent / "depth/depth0.png"
        )
    else:
        np.savetxt(scene.source_images[0].parent.parent / "poses.txt", np.eye(4) * 2)

    with pytest.raises(ValueError, match="staging manifest mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def test_raw_sevenscenes_projected_depth_mutation_invalidates_existing_staging(tmp_path):
    root = tmp_path / "raw-data"
    scene_dir = root / "7-Scenes/chess/seq-03"
    image = _png(scene_dir / "frame-000000.color.png", 2)
    from PIL import Image

    Image.fromarray(np.full((480, 640), 1000, dtype=np.uint16)).save(
        scene_dir / "frame-000000.depth.png"
    )
    Image.fromarray(np.full((480, 640), 1000, dtype=np.uint16)).save(
        scene_dir / "frame-000000.depth.proj.png"
    )
    np.savetxt(scene_dir / "frame-000000.pose.txt", np.eye(4))
    scene = ResolvedScene(
        scene_id="raw-7scenes",
        dataset=DatasetKind.SEVEN_SCENES,
        scene="chess/seq-03",
        slice_id="f000000-000001-s1",
        approved_data_root=root.resolve(),
        source_images=(image,),
        selection=ResolvedFrameSelection(0, 1, 1, (0,)),
        evaluation_kind=EvaluationKind.POINTCLOUD,
        poses_path=None,
        prepared_gt_path=None,
        frame_index_map=None,
        expected_gt_shape=(392, 518),
    )
    stage_scene(scene, tmp_path / "campaign")
    Image.fromarray(np.full((480, 640), 2000, dtype=np.uint16)).save(
        scene_dir / "frame-000000.depth.proj.png"
    )

    with pytest.raises(ValueError, match="staging manifest mismatch"):
        stage_scene(scene, tmp_path / "campaign")


def test_resolve_scene_rejects_short_approved_map_for_nonstandard_slice(tmp_path):
    root = tmp_path / "NeuralRGBD/thin_geometry"
    for frame_id in (0, 10, 20):
        _png(root / "images" / f"img{frame_id}.png", frame_id)
        _png(root / "depth" / f"depth{frame_id}.png", frame_id)
    np.savetxt(root / "poses.txt", np.tile(np.eye(4), (21, 1)))
    sequence_map = tmp_path / "short-map.json"
    sequence_map.write_text(json.dumps({"thin_geometry": [0, 10, 20]}), encoding="utf-8")
    config_path = Path(__file__).resolve().parents[2] / "configs/experiments/window_reference_campaign.yaml"
    loaded = load_campaign_config(config_path, CampaignOverrides(preset="pointcloud-small", data_root=tmp_path))
    scene_config = replace(loaded.config.scenes["nrgbd-thin-geometry"], frame_index_map=str(sequence_map))
    config = replace(
        loaded.config,
        scenes={**loaded.config.scenes, "nrgbd-thin-geometry": scene_config},
        selected_scenes=(PresetSceneConfig("nrgbd-thin-geometry", 0, 3, 1),),
    )

    with pytest.raises(ValueError, match="exactly 40 mapped frames"):
        resolve_scene(config, config.selected_scenes[0])


def test_resolve_scene_rejects_physical_kitti_24_column_pose_row(tmp_path):
    image_dir = tmp_path / "KITTI/04/image_2"
    _png(image_dir / "0.png")
    _png(image_dir / "1.png")
    poses = image_dir.parent / "poses.txt"
    poses.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(poses, np.arange(24, dtype=float).reshape(1, 24))
    loaded = load_campaign_config(
        Path(__file__).resolve().parents[2] / "configs/experiments/window_reference_campaign.yaml",
        CampaignOverrides(preset="kitti-smoke", data_root=tmp_path),
    )

    with pytest.raises(ValueError, match="12 values per physical row"):
        resolve_scene(loaded.config, loaded.config.selected_scenes[0])


def test_staging_rejects_physical_kitti_24_column_pose_row(tmp_path):
    root = tmp_path / "data"
    images = tuple(_png(root / "04/image_2" / f"{index:06d}.png", index) for index in range(2))
    np.savetxt(root / "04/poses.txt", np.arange(24, dtype=float).reshape(1, 24))
    scene = ResolvedScene(
        scene_id="kitti-bad-pose",
        dataset=DatasetKind.KITTI,
        scene="04",
        slice_id="f000000-000002-s1",
        approved_data_root=root.resolve(),
        source_images=images,
        selection=ResolvedFrameSelection(0, 2, 1, (0, 1)),
        evaluation_kind=EvaluationKind.INTERNAL_TRAJECTORY,
        poses_path=(root / "04/poses.txt").resolve(),
        prepared_gt_path=None,
        frame_index_map=None,
        expected_gt_shape=None,
    )

    with pytest.raises(ValueError, match="12 values per physical row"):
        stage_scene(scene, tmp_path / "campaign")


def test_raw_dataset_output_requires_pointmap_leading_dimension_to_match_ids(tmp_path):
    scene = _pointcloud_scene(tmp_path, ids=(0, 10, 20), requested=(10, 20))
    data = {
        "ind": np.array([10, 20], dtype=np.int64),
        "image_paths": [str(path) for path in scene.source_images],
        "pointclouds": np.zeros((1, 392, 518, 3), dtype=np.float32),
        "valid_mask": np.ones((1, 392, 518), dtype=bool),
    }

    with pytest.raises(ValueError, match="leading dimension"):
        _validate_raw_dataset_output(scene, data)


def test_publish_race_never_clobbers_concurrent_invalid_final(tmp_path, monkeypatch):
    scene = _kitti_scene(tmp_path)
    root = (tmp_path / "campaign").resolve()
    final = root / "prepared/kitti" / scene.slice_id
    original_mkdir = staging_module.os.mkdir

    def race_mkdir(path, mode=0o777, *args, **kwargs):
        if Path(path) == final:
            original_mkdir(path, mode, *args, **kwargs)
            (final / "sentinel").write_text("winner", encoding="utf-8")
            raise FileExistsError(path)
        return original_mkdir(path, mode, *args, **kwargs)

    monkeypatch.setattr(staging_module.os, "mkdir", race_mkdir)
    with pytest.raises(ValueError, match="staging"):
        stage_scene(scene, root)
    assert (final / "sentinel").read_text(encoding="utf-8") == "winner"
    assert not (final / "images").exists()


def test_owned_path_helpers_reject_escape_and_only_remove_owned_descendants(tmp_path):
    owned = tmp_path / "campaign/prepared"
    child = owned / "scene"
    child.mkdir(parents=True)
    assert require_descendant(child, owned) == child.resolve()
    with pytest.raises(ValueError, match="outside"):
        require_descendant(tmp_path / "outside", owned)
    with pytest.raises(ValueError, match="outside"):
        guarded_remove(tmp_path / "outside", owned)
    guarded_remove(child, owned)
    assert not child.exists()
