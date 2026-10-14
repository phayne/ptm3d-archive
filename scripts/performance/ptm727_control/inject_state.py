#!/usr/bin/env python3
"""PTM-727 decisive control: build the SPUN-UP state file by patching an engine-written
donor state (thermal_state_io.h v2) with Fermi's replay spun-up columns.

  donor   : the engine's own end-of-window column at the control's start epoch
            (configs/donor_dt65.yaml: one output step ending exactly there, look-ahead 1)
  replay  : spun_L1start_<tag>.npz from replay_spun_state.py -- full T(z) at the L1-start
            phase for the 93,563 faces whose T_max < 130 K in some lunation (Fermi's subset;
            every other face is lit and spins up within the equilibration)
  out     : the donor file with those faces' columns replaced (replay grid -> the donor's
            own depth grid, read from the v2 trailer, linear in z), every other byte kept,
            CRC32 recomputed. The epoch / mesh-hash / config-identity fields are the donor's,
            so the engine's warm-start checks apply unchanged.

  inject_state.py donor_state.bin spun_L1start_f093.npz spun_state.bin
"""
import struct
import sys
import zlib

import numpy as np

MAGIC, TMAGIC = 0x50544D54, 0x50544D58


def read_state(path):
    b = bytearray(open(path, "rb").read())
    magic, nf, nl = struct.unpack_from("<III", b, 0)
    if magic != MAGIC:
        raise SystemExit(f"FATAL: {path} is not a PTM state file (magic {magic:#x})")
    body = 12 + 4 * nf * nl
    if len(b) <= body:
        raise SystemExit(f"FATAL: {path} is a v1 file (no trailer): no depth grid / epoch to check")
    tmagic, ver, tbytes, nl2 = struct.unpack_from("<IIII", b, body)
    if tmagic != TMAGIC or nl2 != nl or body + tbytes != len(b):
        raise SystemExit(f"FATAL: {path}: malformed v2 trailer")
    z = np.frombuffer(bytes(b[body + 16: body + 16 + 8 * nl]), "<f8").copy()
    et, cadence = struct.unpack_from("<dd", b, body + 16 + 8 * nl)
    crc_stored = struct.unpack_from("<I", b, len(b) - 4)[0]
    if zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF != crc_stored:
        raise SystemExit(f"FATAL: {path}: CRC32 mismatch on the donor itself")
    T = np.frombuffer(bytes(b[12:body]), "<f4").reshape(nl, nf).copy()
    return b, nf, nl, z, et, cadence, T


def main():
    donor, replay, out = sys.argv[1:4]
    b, nf, nl, z, et, cad, T = read_state(donor)
    R = np.load(replay)
    idx, zr, Tr = R["idx"], R["z"], R["T"]           # Tr: (NZ_replay, n_idx)
    if idx.max() >= nf:
        raise SystemExit(f"FATAL: replay face index {idx.max()} >= donor faces {nf}")
    if z[-1] > zr[-1] * 1.01 or z[0] < zr[0] - 1e-9:
        raise SystemExit(f"FATAL: donor grid z in [{z[0]}, {z[-1]}] is outside the replay grid "
                         f"[{zr[0]}, {zr[-1]}]; refusing to extrapolate a column")
    # linear interpolation in z, same weights for every face
    j = np.clip(np.searchsorted(zr, z, side="right") - 1, 0, len(zr) - 2)
    w = ((z - zr[j]) / (zr[j + 1] - zr[j]))[:, None]
    Tnew = (1 - w) * Tr[j] + w * Tr[j + 1]           # (nl, n_idx)
    before = T[:, idx].astype(np.float64)
    T[:, idx] = Tnew.astype(np.float32)
    body = 12 + 4 * nf * nl
    b[12:body] = T.astype("<f4").tobytes()
    struct.pack_into("<I", b, len(b) - 4, zlib.crc32(bytes(b[:-4])) & 0xFFFFFFFF)
    open(out, "wb").write(b)
    d = Tnew - before
    print(f"[inject] donor {donor}: {nf} faces x {nl} layers, ET {et:.3f}, cadence {cad:.3f} s")
    print(f"[inject] patched {idx.size} faces from {replay}; replay grid {len(zr)} nodes -> donor {nl}")
    for k in (0, nl // 3, 2 * nl // 3, nl - 1):
        print(f"[inject]   z={z[k]:.4f} m: donor median {np.median(before[k]):.2f} K -> spun "
              f"{np.median(Tnew[k]):.2f} K (median change {np.median(d[k]):+.2f} K)")
    # read back: the engine's own checks must pass on the written file
    read_state(out)
    print(f"[inject] wrote {out} (CRC re-verified)")


if __name__ == "__main__":
    main()
