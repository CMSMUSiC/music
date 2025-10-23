import os
import subprocess as sp
import sys
from typing import Literal, Sequence, Union


StreamMode = Literal["auto", "lines", "chars"]


def run_stream_shell(
    cmd: Union[str, Sequence[str]],
    *,
    cwd: str | None = None,
    env: dict | None = None,
    shell_exe: str | None = "/bin/bash",  # POSIX shell to run under
    merge_stderr: bool = True,  # show progress bars printed to stderr
    stream_mode: StreamMode = "auto",  # "auto" picks "chars" if merge_stderr else "lines"
    line_buffer_hint: bool = True,  # add stdbuf (POSIX) to coax timely flushing
) -> int:
    """
    Run a command *via a shell* and stream output as it appears.

    - If stream_mode="chars", reads raw bytes and writes them through (best for progress bars).
    - If stream_mode="lines", reads text lines (nice for typical logs).
    - "auto": uses "chars" when merge_stderr=True (common for progress bars), else "lines".
    """
    # Normalize to a single command string for the shell
    if not isinstance(cmd, str):
        import shlex

        cmd = " ".join(shlex.quote(str(c)) for c in cmd)

    # Heuristic: if user merges stderr (likely progress bars), prefer char streaming
    if stream_mode == "auto":
        stream_mode = "chars" if merge_stderr else "lines"

    # Reduce buffering of the child (POSIX)
    if line_buffer_hint and os.name == "posix":
        # For progress bars, unbuffered (-o0 -e0) makes \r updates snappy
        if stream_mode == "chars":
            cmd = f"stdbuf -o0 -e0 {cmd}"
        else:
            cmd = f"stdbuf -oL -eL {cmd}"

    # Build Popen kwargs
    popen_kwargs = dict(
        cwd=cwd,
        env=env,
        shell=True,
    )
    if os.name == "posix" and shell_exe:
        popen_kwargs["executable"] = shell_exe

    # Choose text/binary mode based on streaming style
    if stream_mode == "chars":
        popen_kwargs.update(
            stdout=sp.PIPE,
            stderr=sp.STDOUT if merge_stderr else sp.PIPE,
            text=False,
            bufsize=0,
        )  # binary, unbuffered
    else:  # "lines"
        popen_kwargs.update(
            stdout=sp.PIPE,
            stderr=sp.STDOUT if merge_stderr else sp.PIPE,
            text=True,
            bufsize=1,
        )  # text, line-buffered

    # Launch
    proc = sp.Popen(cmd, **popen_kwargs)

    try:
        if merge_stderr:
            # Single stream path
            assert proc.stdout is not None
            if stream_mode == "chars":
                out = proc.stdout
                w = sys.stdout.buffer
                for chunk in iter(lambda: out.read(1024), b""):
                    w.write(chunk)
                    w.flush()
            else:
                for line in proc.stdout:
                    print(line, end="")
        else:
            # Dual-stream path
            import queue
            import threading

            q: "queue.Queue[tuple[str, bytes|str|None]]" = queue.Queue()

            def pump_bytes(stream, tag):
                for chunk in iter(lambda: stream.read(1024), b""):
                    q.put((tag, chunk))
                q.put((tag, None))

            def pump_lines(stream, tag):
                for line in iter(stream.readline, ""):
                    q.put((tag, line))
                q.put((tag, None))

            if stream_mode == "chars":
                t_out = threading.Thread(
                    target=pump_bytes, args=(proc.stdout, "out"), daemon=True
                )
                t_err = threading.Thread(
                    target=pump_bytes, args=(proc.stderr, "err"), daemon=True
                )
            else:
                t_out = threading.Thread(
                    target=pump_lines, args=(proc.stdout, "out"), daemon=True
                )
                t_err = threading.Thread(
                    target=pump_lines, args=(proc.stderr, "err"), daemon=True
                )

            t_out.start()
            t_err.start()

            done = {"out": False, "err": False}
            while not all(done.values()):
                tag, payload = q.get()
                if payload is None:
                    done[tag] = True
                    continue
                if stream_mode == "chars":
                    buf = sys.stdout.buffer
                    if tag == "err":
                        # minimal prefixing without breaking \r animations too much
                        buf.write(b"[stderr] ")
                    buf.write(payload)
                    buf.flush()
                else:
                    if tag == "err":
                        print(f"[stderr] {payload}", end="")
                    else:
                        print(payload, end="")

            t_out.join()
            t_err.join()
    except KeyboardInterrupt:
        # Forward Ctrl-C to the whole group on POSIX
        try:
            if os.name == "posix":
                import signal

                os.killpg(proc.pid, signal.SIGINT)
        except Exception:
            pass
        finally:
            proc.wait()
    finally:
        return proc.wait()
