"""Artifact layout, checksum, and load-instruction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from distillery.training.artifacts import (
    RUN_ARTIFACT_LAYOUT,
    build_run_artifact_layout,
    generate_load_instructions,
    parse_sha256sums,
    sha256_file,
    verify_sha256sums,
    write_load_test_document,
    write_sha256sums,
)

REVISION = "a" * 40


def test_run_layout_contains_required_paths() -> None:
    layout = build_run_artifact_layout(
        run_id="run_abc",
        root_prefix="s3://bucket/runs/run_abc",
    )
    assert layout.resolve("manifest") == "s3://bucket/runs/run_abc/manifest.json"
    assert "model/adapter" in RUN_ARTIFACT_LAYOUT["model_adapter_dir"]
    assert any(f.relative_path == "integrity/SHA256SUMS" for f in layout.files)


def test_sha256sums_roundtrip_and_verify(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "nested" / "b.txt"
    f2.parent.mkdir()
    f1.write_text("hello\n", encoding="utf-8")
    f2.write_text("world\n", encoding="utf-8")
    entries = {
        "a.txt": sha256_file(f1),
        "nested/b.txt": sha256_file(f2),
    }
    sums_path = write_sha256sums(entries, destination=tmp_path / "SHA256SUMS")
    parsed = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    assert parsed == entries
    assert verify_sha256sums(entries, root=tmp_path) == ()

    f1.write_text("tampered\n", encoding="utf-8")
    violations = verify_sha256sums(entries, root=tmp_path)
    assert "mismatch:a.txt" in violations


def test_load_instructions_mention_peft_and_base() -> None:
    text = generate_load_instructions(
        student_base_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision=REVISION,
        adapter_uri="s3://bucket/runs/run_x/model/adapter",
        merged_uri="s3://bucket/runs/run_x/model/merged",
    )
    assert "PeftModel.from_pretrained" in text
    assert "Qwen/Qwen2.5-0.5B-Instruct" in text
    assert REVISION in text
    assert "merged" in text


def test_write_load_test_document(tmp_path: Path) -> None:
    path = write_load_test_document(
        tmp_path / "load_test.json",
        student_base_id="Qwen/Qwen2.5-0.5B-Instruct",
        student_revision=REVISION,
        adapter_uri="/tmp/adapter",
        checksums={"adapter_model.safetensors": "a" * 64},
    )
    assert path.is_file()
    content = path.read_text(encoding="utf-8")
    assert "distillery.load_test.v1" in content
    assert "load_instructions" in content


@pytest.mark.parametrize(
    "relative_path",
    [
        "/etc/passwd",
        "../escape",
        "nested/../../escape",
        "nested\\windows-escape",
        "nested//empty",
        "./dot",
    ],
)
def test_checksum_parser_rejects_unsafe_paths(relative_path: str) -> None:
    with pytest.raises(ValueError, match="unsafe checksum path"):
        parse_sha256sums(f"{'a' * 64}  {relative_path}\n")


def test_checksum_parser_rejects_duplicates_and_malformed_digests() -> None:
    duplicate = f"{'a' * 64}  a.txt\n{'b' * 64}  a.txt\n"
    with pytest.raises(ValueError, match="duplicate"):
        parse_sha256sums(duplicate)
    for digest in ("a" * 63, "A" * 64, "not-a-digest"):
        with pytest.raises(ValueError, match="malformed"):
            parse_sha256sums(f"{digest}  a.txt\n")


def test_verify_rejects_symlinks_nonregular_files_and_escapes(tmp_path: Path) -> None:
    regular = tmp_path / "regular.txt"
    regular.write_text("regular\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(regular)
    directory = tmp_path / "directory"
    directory.mkdir()

    assert verify_sha256sums({"link.txt": sha256_file(regular)}, root=tmp_path) == (
        "symlink:link.txt",
    )
    assert verify_sha256sums({"directory": "a" * 64}, root=tmp_path) == (
        "non_regular:directory",
    )
    violations = verify_sha256sums({"../escape": "a" * 64}, root=tmp_path)
    assert len(violations) == 1
    assert violations[0].startswith("invalid_entry:../escape:")


def test_layout_resolve_rejects_traversal() -> None:
    layout = build_run_artifact_layout(
        run_id="run_abc",
        root_prefix="s3://bucket/runs/run_abc",
    )
    with pytest.raises(ValueError, match="unsafe checksum path"):
        layout.resolve("../../escape")
