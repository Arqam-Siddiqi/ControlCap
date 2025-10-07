#!/usr/bin/env python
# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help
from __future__ import division

import atexit
import logging
import os
import re
import subprocess
import sys
import threading
import time
import psutil

METEOR_JAR = 'meteor-1.5.jar'


def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


class Meteor:
    def __init__(self):
        self.lock = threading.Lock()
        self.debug = os.environ.get("METEOR_DEBUG", "0") == "1"

        mem = '2G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("Less than 2GB RAM available, reducing METEOR heap to 1G.")
            mem = '1G'

        jar_dir = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(jar_dir, METEOR_JAR)

        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"[METEOR] JAR not found at {jar_path}")

        meteor_cmd = [
            'java',
            f'-Xmx{mem}',
            '-jar',
            METEOR_JAR,
            '-', '-', '-stdio', '-l', 'en', '-norm'
        ]

        env = os.environ.copy()
        env['LC_ALL'] = "C"

        self.meteor_p = subprocess.Popen(
            meteor_cmd,
            cwd=jar_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(0.2)
        if self.meteor_p.poll() is not None:
            stderr_out = self.meteor_p.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(
                f"[METEOR] Process exited immediately (code={self.meteor_p.returncode}). "
                f"stderr:\\n{stderr_out}"
            )

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if getattr(self, "meteor_p", None):
                try:
                    if self.meteor_p.stdin:
                        try:
                            self.meteor_p.stdin.write(enc('QUIT\\n'))
                            self.meteor_p.stdin.flush()
                        except Exception:
                            pass
                    self.meteor_p.kill()
                    self.meteor_p.wait(timeout=1)
                except Exception:
                    pass
                self.meteor_p = None
        try:
            if atexit is not None and atexit.unregister is not None:
                atexit.unregister(self.close)
        except Exception:
            pass

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        imgIds = list(gts.keys())
        scores = []
        eval_line = 'EVAL'

        with self.lock:
            for i in imgIds:
                assert len(res[i]) == 1
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            try:
                self._safe_write(eval_line + '\\n')
            except BrokenPipeError as e:
                self._raise_with_stderr("Broken pipe during EVAL write", e)

            for _ in range(len(imgIds)):
                v = self.meteor_p.stdout.readline()
                if not v:
                    self._raise_with_stderr("No per-image score line (stdout closed).")
                try:
                    scores.append(float(dec(v.strip())))
                except Exception:
                    sys.stderr.write(f"[METEOR] Bad score line: {v}\\n")
                    self._raise_with_stderr("Failed parsing score line.")

            final_line = self.meteor_p.stdout.readline()
            if not final_line:
                self._raise_with_stderr("No final aggregate line (stdout closed).")
            try:
                score = float(dec(final_line.strip()))
            except Exception:
                self._raise_with_stderr(f"Failed parsing final score line: {final_line!r}")

        return score, scores

    def _safe_write(self, line: str):
        if self.meteor_p.poll() is not None:
            self._raise_with_stderr("Process already exited before write.")
        self.meteor_p.stdin.write(enc(line))
        self.meteor_p.stdin.flush()

    def _raise_with_stderr(self, msg, original_exc=None):
        rc = self.meteor_p.returncode
        try:
            stderr_out = self.meteor_p.stderr.read().decode('utf-8', errors='ignore')
        except Exception:
            stderr_out = ""
        full = (
            f"[METEOR ERROR] {msg}. returncode={rc}\\n"
            f"--- STDERR ---\\n{stderr_out}\\n---------------"
        )
        if original_exc:
            raise RuntimeError(full) from original_exc
        raise RuntimeError(full)

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace('|||', '').strip()
        reference_list = [r.replace('|||', '').strip() for r in reference_list if r.strip() != ""]
        if len(reference_list) == 0:
            return "0 0 0"
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = re.sub(r'\\s+', ' ', score_line)
        try:
            self._safe_write(f"{score_line}\\n")
        except BrokenPipeError as e:
            self._raise_with_stderr("Broken pipe during SCORE write", e)
        out = self.meteor_p.stdout.readline()
        if not out:
            self._raise_with_stderr("No SCORE response line (stdout closed).")
        return dec(out).strip()

    def __del__(self):
        self.close()