"""Descriptor-relative atomic filesystem operations for snapshots."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from uuid import uuid4


class SnapshotFiles:
    """Own symlink-safe file and directory operations beneath one store root."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir

    def _write_atomic(self, destination: Path, content: bytes) -> None:
        relative = self._relative_to_root(destination)
        directory = self._open_directory(relative.parent, create=True)
        temporary = f".{relative.name}.{uuid4().hex}.tmp"
        primary_error: BaseException | None = None
        try:
            descriptor = self._open_file(
                directory,
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                self._reject_symlink(directory, relative.name)
            except FileNotFoundError:
                pass
            os.replace(
                temporary,
                relative.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
            )
            self._fsync_directory(directory)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"temporary snapshot cleanup failed: {cleanup_error}")
            os.close(directory)

    def _read_file(self, relative: Path) -> bytes | None:
        directory = self._open_directory(relative.parent, create=True)
        try:
            try:
                self._reject_symlink(directory, relative.name)
            except FileNotFoundError:
                return None
            descriptor = self._open_file(directory, relative.name, os.O_RDONLY)
            with os.fdopen(descriptor, "rb") as input_file:
                return input_file.read()
        finally:
            os.close(directory)

    def _replace_atomic(self, source: Path, destination: Path) -> None:
        source_relative = self._relative_to_root(source)
        destination_relative = self._relative_to_root(destination)
        source_directory = self._open_directory(source_relative.parent, create=False)
        destination_directory = self._open_directory(destination_relative.parent, create=True)
        try:
            self._reject_symlink(source_directory, source_relative.name)
            try:
                self._reject_symlink(destination_directory, destination_relative.name)
            except FileNotFoundError:
                pass
            os.replace(
                source_relative.name,
                destination_relative.name,
                src_dir_fd=source_directory,
                dst_dir_fd=destination_directory,
            )
            self._fsync_directory(destination_directory)
            if source_directory != destination_directory:
                self._fsync_directory(source_directory)
        finally:
            os.close(destination_directory)
            os.close(source_directory)

    def _unlink_file(self, relative: Path) -> bool:
        directory = self._open_directory(relative.parent, create=True)
        try:
            try:
                self._reject_symlink(directory, relative.name)
                os.unlink(relative.name, dir_fd=directory)
            except FileNotFoundError:
                return False
            self._fsync_directory(directory)
            return True
        finally:
            os.close(directory)

    def _open_directory(self, relative: Path, *, create: bool) -> int:
        descriptor = self._open_store_root()
        descriptors = [descriptor]
        try:
            for component in relative.parts:
                _validate_path_component(component)
                try:
                    self._reject_symlink(descriptor, component)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                descriptors.append(child)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
        else:
            try:
                for opened in reversed(descriptors):
                    self._fsync_directory(opened)
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
        finally:
            for opened in descriptors[:-1]:
                os.close(opened)

    def _open_store_root(self) -> int:
        try:
            root_stat = os.lstat(self.store_dir)
        except FileNotFoundError:
            try:
                self.store_dir.mkdir(mode=0o700)
            except FileExistsError:
                pass
            root_stat = os.lstat(self.store_dir)
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"snapshot store root is not a directory: {self.store_dir}")
        return os.open(self.store_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    @staticmethod
    def _open_file(directory: int, name: str, flags: int, mode: int = 0o600) -> int:
        return os.open(name, flags | os.O_NOFOLLOW, mode, dir_fd=directory)

    @staticmethod
    def _reject_symlink(directory: int, name: str) -> None:
        entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if stat.S_ISLNK(entry.st_mode):
            raise ValueError(f"snapshot path component is a symlink: {name}")
        if not stat.S_ISREG(entry.st_mode) and not stat.S_ISDIR(entry.st_mode):
            raise ValueError(f"snapshot path component has unsupported type: {name}")

    def _relative_to_root(self, path: Path) -> Path:
        try:
            relative = path.relative_to(self.store_dir)
        except ValueError as exc:
            raise ValueError(f"snapshot path escapes store root: {path}") from exc
        if not relative.parts:
            raise ValueError("snapshot destination cannot be the store root")
        return relative

    @staticmethod
    def _fsync_directory(descriptor: int) -> None:
        os.fsync(descriptor)


def _validate_path_component(component: str) -> None:
    if component in ("", ".", ".."):
        raise ValueError(f"invalid snapshot path component: {component!r}")


__all__ = ["SnapshotFiles"]
