"""Unchanged experiment worker with periodic stacks for stalled native calls."""
import faulthandler
import signal
from native_boundary_worker import main

if __name__ == '__main__':
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
