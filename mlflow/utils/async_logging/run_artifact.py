import os
import shutil
import tempfile
import threading
from typing import TYPE_CHECKING, Optional, Union

from mlflow import MlflowException

if TYPE_CHECKING:
    import PIL


class RunArtifact:
    def __init__(
        self,
        artifact_path: str,
        completion_event: threading.Event,
        local_file: Optional[str] = None,
        filename: Optional[str] = None,
        artifact: Optional[Union["PIL.Image.Image",]] = None,
    ) -> None:
        """Initializes an instance of `RunArtifacts`.

        Args:
            filename: Filename of the artifact to be logged
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
            artifact: The artifact to be logged.
            completion_event: A threading.Event object.
        """
        self.filename = filename
        self.artifact_path = artifact_path
        self.artifact = artifact
        self.completion_event = completion_event
        self.tmp_dir = None
        self._local_file = local_file
        self._exception = None

    def __enter__(self) -> str:
        """Creates a temporary directory for artifact storage if required."""
        if self.artifact is not None:
            self.tmp_dir = tempfile.mkdtemp()
        return self._get_local_filepath()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up the temporary directory when exiting context."""
        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _get_local_filepath(self):
        if self.artifact is not None and self.filename is not None:
            import PIL  # PIL should be available if we get here

            if isinstance(self.artifact, PIL.Image.Image):
                self._local_file = os.path.join(self.tmp_dir, self.filename)
                self.artifact.save(self._local_file)
            else:
                raise MlflowException(
                    "Unsupported artifact type. Only PIL.Image.Image and local_file are supported for async artifacts."
                )
        if not self._local_file:
            raise MlflowException("No artifact or local_file provided for logging.")

        return self._local_file

    @property
    def exception(self):
        """Exception raised during logging the batch."""
        return self._exception

    @exception.setter
    def exception(self, exception):
        self._exception = exception
