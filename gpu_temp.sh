# Sourced by the run_*.sh queues for their cooldown pauses.
#
# gpu_temp prints the GPU temperature in whole degrees C, or nothing when no
# reading is available, in which case the queues never pause. NVIDIA GPUs are
# read through nvidia-smi; AMD GPUs through the amdgpu driver's edge sensor in
# sysfs, which needs no ROCm tools. The two vendors' sensors are not the same
# measurement, so set COOL_ABOVE for the machine rather than carrying it over.

gpu_temp () {
  if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -n 1
    return
  fi
  local h
  for h in /sys/class/hwmon/hwmon*; do
    if [ "$(cat "$h/name" 2> /dev/null)" = amdgpu ] && [ -r "$h/temp1_input" ]; then
      echo $(( $(cat "$h/temp1_input") / 1000 ))
      return
    fi
  done
}
