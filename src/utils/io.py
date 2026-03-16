"""
Asynchronous IO utilities.

Designed for GPU-heavy NeuroAI / RL workloads:
- Handles heavy disk I/O (checkpoints, arrays) in background threads.
- Forces explicit GPU->CPU synchronization on the main thread to avoid CUDA context errors.
- Prevents memory leaks by pruning completed futures.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Optional, Union, Dict, Any, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import matplotlib.figure

log = logging.getLogger(__name__)


class AsyncIOManager:
    """
    Centralized manager for asynchronous disk IO.
    
    Usage:
        with AsyncIOManager() as io:
            io.save_numpy_async("data.npy", array)
    """

    def __init__(self, max_workers: int = 2, strict: bool = False):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future] = []
        self.strict = strict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _prune_futures(self):
        """Removes completed futures to prevent memory leaks in long runs."""
        # Keep only running or pending futures
        self._futures = [f for f in self._futures if not f.done()]

    def _track(self, fut: Future) -> Future:
        """Schedules a future and performs periodic cleanup."""
        self._futures.append(fut)
        if len(self._futures) > 100:
            self._prune_futures()
        return fut

    def _handle_exception(self, exc: Exception, context: str):
        msg = f"AsyncIO error in {context}: {exc}"
        if self.strict:
            log.error(msg)
            raise exc
        log.warning(msg)

    @staticmethod
    def _prepare_path(path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Numpy
    # ------------------------------------------------------------------

    @staticmethod
    def _save_numpy_worker(path: Path, arr: np.ndarray, allow_pickle: bool, fix_imports: bool):
        try:
            np.save(path, arr, allow_pickle=allow_pickle, fix_imports=fix_imports)
        except Exception as e:
            raise IOError(f"Failed to save numpy array to {path}") from e

    def save_numpy_async(
        self,
        path: Union[str, Path],
        array: np.ndarray,
        *,
        allow_pickle: bool = True,
        fix_imports: bool = True,
    ) -> Future:
        p = self._prepare_path(path)
        fut = self._executor.submit(
            self._save_numpy_worker, p, array, allow_pickle, fix_imports
        )
        return self._track(fut)

    def save_tensor_as_numpy_async(
        self,
        path: Union[str, Path],
        tensor: torch.Tensor,
        *,
        allow_pickle: bool = True,
        fix_imports: bool = True,
    ) -> Future:
        """
        Converts Tensor -> Numpy on the main thread (blocking), then saves async.
        Crucial for safety with CUDA tensors.
        """
        try:
            # Synchronize and copy to CPU immediately
            arr = tensor.detach().cpu().numpy()
        except Exception as e:
            self._handle_exception(e, "tensor_to_numpy")
            # Return a dummy future to satisfy type signature
            f = Future()
            f.set_exception(e)
            return f

        return self.save_numpy_async(
            path, arr, allow_pickle=allow_pickle, fix_imports=fix_imports
        )

    # ------------------------------------------------------------------
    # Matplotlib
    # ------------------------------------------------------------------

    @staticmethod
    def _save_figure_worker(path: Path, fig: 'matplotlib.figure.Figure', close: bool):
        try:
            import matplotlib.pyplot as plt
            fig.tight_layout(pad=2.0)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            if close:
                plt.close(fig)
        except Exception as e:
            raise IOError(f"Failed to save figure to {path}") from e

    def save_figure_async(
        self, 
        path: Union[str, Path], 
        fig: 'matplotlib.figure.Figure',
        close_after_save: bool = True
    ) -> Future:
        p = self._prepare_path(path)
        # Note: Matplotlib is not thread-safe. Ideally, use non-interactive backends.
        fut = self._executor.submit(self._save_figure_worker, p, fig, close_after_save)
        return self._track(fut)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _save_ckpt_worker(path: Path, payload: Dict[str, Any]):
        # PyTorch save is thread-safe for CPU tensors/dicts
        torch.save(payload, path)

    def save_checkpoint_async(self, path: Union[str, Path], checkpoint: Dict[str, Any]) -> Future:
        p = self._prepare_path(path)
        fut = self._executor.submit(self._save_ckpt_worker, p, checkpoint)
        return self._track(fut)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wait(self):
        """Block until all currently scheduled futures are done."""
        for f in as_completed(self._futures):
            if f.exception() and self.strict:
                raise f.exception()

    def flush(self):
        """Wait for tasks and clear the tracker."""
        self.wait()
        self._futures.clear()

    def close(self):
        """Shutdown the executor."""
        self.flush()
        self._executor.shutdown(wait=True)








