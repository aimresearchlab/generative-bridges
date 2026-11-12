"""Which device to run on, and which GPU stack drives it.

PyTorch drives an AMD GPU under ROCm through the same torch.cuda API and the
same "cuda" device type as an NVIDIA GPU, so the training, sampling and
evaluation code has no vendor branches: "cuda" names whichever GPU the
installed PyTorch build was compiled for, and the torch.backends.cudnn
switches the trainers set apply to MIOpen under ROCm.

The device is the first GPU PyTorch sees. BRIDGES_DEVICE overrides it, for
example BRIDGES_DEVICE=cuda:1 for a second GPU, or BRIDGES_DEVICE=cpu to run a
small check on a machine without one. Without that override a missing GPU is
an error, so a broken GPU setup stops at launch instead of quietly training on
the CPU.

    python -m bridges.device             # report the stack and run a short check
    python -m bridges.device --compile   # the same check through torch.compile
"""

from __future__ import annotations

import argparse, os, sys, time

import torch


def _why_no_gpu() -> str:
    if torch.version.hip:
        return ("This is the ROCm build of PyTorch. Check that /dev/kfd and /dev/dri are "
                "readable (membership of the render and video groups) and that the "
                "device package for this GPU is installed (see REPRODUCE.md).")
    if torch.version.cuda:
        if os.path.exists("/dev/kfd"):
            return ("This is the CUDA build of PyTorch, but the machine has an AMD GPU: "
                    "install the ROCm build instead (see REPRODUCE.md).")
        return "This is the CUDA build of PyTorch. Check the NVIDIA driver with nvidia-smi."
    return "This PyTorch build has no GPU support. Install a CUDA or ROCm build (see REPRODUCE.md)."


def default_device() -> str:
    """BRIDGES_DEVICE if set, otherwise "cuda", which is the GPU on NVIDIA and AMD alike."""
    env = os.environ.get("BRIDGES_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("PyTorch sees no GPU. " + _why_no_gpu()
                       + " Set BRIDGES_DEVICE=cpu to run on the CPU anyway.")


def device_type(device) -> str:
    """The device type autocast expects: "cuda" for any GPU, "cpu" for the CPU."""
    return torch.device(device).type


def describe(device) -> str:
    """One line naming the device, its architecture and the GPU stack, for run logs."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return f"device {dev}, torch {torch.__version__}"
    p = torch.cuda.get_device_properties(dev)
    if torch.version.hip:
        arch, stack = p.gcnArchName, f"ROCm (HIP {torch.version.hip})"
    else:
        arch, stack = f"sm_{p.major}{p.minor}", f"CUDA {torch.version.cuda}"
    return (f"device {dev}: {p.name}, {arch}, {p.total_memory / 2**30:.0f} GiB, "
            f"torch {torch.__version__}, {stack}")


def main():
    p = argparse.ArgumentParser(description="Report the device and check that a training "
                                            "step and a short sampling run work on it.")
    p.add_argument("--compile", action="store_true", help="run the check through torch.compile")
    a = p.parse_args()

    from .bridge import BridgeConfig, GaussianBridge
    from .unet import build_unet

    try:
        dev = default_device()
    except RuntimeError as e:
        sys.exit(str(e))
    print(describe(dev), flush=True)

    torch.backends.cudnn.benchmark = True
    model = build_unet("cifar10").to(dev)
    net = torch.compile(model) if a.compile else model
    bridge = GaussianBridge(BridgeConfig(param="drift"))
    y = torch.rand(8, 3, 32, 32, device=dev) * 2 - 1

    t0 = time.time()
    with torch.autocast(device_type(dev), dtype=torch.bfloat16):
        loss, _ = bridge.training_loss(net, y)
    loss.backward()
    with torch.no_grad(), torch.autocast(device_type(dev), dtype=torch.bfloat16):
        x = bridge.sample(net, tuple(y.shape), dev, n_steps=2, sampler="euler")
    ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(x).all())
    print(f"{'ok' if ok else 'FAILED'}: bf16 training step and 2-step sampling of the "
          f"CIFAR-10 UNet{' under torch.compile' if a.compile else ''} "
          f"in {time.time() - t0:.0f} s", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
