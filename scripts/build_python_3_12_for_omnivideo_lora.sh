#!/usr/bin/env bash
# Build an isolated CPython for the Megatron-SWIFT Qwen3-Omni LoRA environment.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_version="${LORA_PYTHON_VERSION:-3.12.13}"
runtime_root="${LORA_RUNTIME_DIR:-/tmp/omnivideo_100k_qwen3_omni_lora}"
build_root="${LORA_PYTHON_ROOT:-${runtime_root}/python-${python_version}}"
python_bin="${build_root}/bin/python3.12"
archive="${build_root}/Python-${python_version}.tgz"
source_dir="${build_root}/Python-${python_version}"
build_log="${build_root}/build.log"
deps_root="${runtime_root}/python-build-deps"
local_ffi_root="${deps_root}/libffi-dev"
sysroot="${deps_root}/sysroot"
sysroot_include="${sysroot}/usr/include"
sysroot_multiarch_include="${sysroot_include}/x86_64-linux-gnu"
sysroot_lib_dir="${sysroot}/usr/lib/x86_64-linux-gnu"
required_modules=(bz2 ctypes lzma sqlite3 readline uuid curses ssl zlib)
module_imports="$(IFS=,; printf '%s' "${required_modules[*]}")"

if [[ -x "${python_bin}" ]] && "${python_bin}" -c "import ${module_imports}"; then
  "${python_bin}" --version
  exit 0
fi

# ``build_log`` is also used while staging build dependencies below, so the
# destination directory must exist before the first redirected command.
mkdir -p "${build_root}"

# The worker image only has the runtime libraries.  Stage the development
# packages under /tmp so CPython can build its optional standard modules
# without changing the host image or requiring sudo.
if [[ ! -f "${sysroot_include}/bzlib.h" ]] || [[ ! -f "${sysroot_include}/lzma.h" ]] || \
   [[ ! -f "${sysroot_include}/sqlite3.h" ]] || [[ ! -f "${sysroot_include}/readline/readline.h" ]] || \
   [[ ! -f "${sysroot_include}/curses.h" ]]; then
  deb_root="${deps_root}/debs"
  mkdir -p "${deb_root}" "${sysroot}"
  build_packages=(
    libbz2-dev liblzma-dev libsqlite3-dev libreadline-dev libgdbm-dev
    libgdbm-compat-dev libncurses-dev uuid-dev zlib1g-dev libssl-dev libexpat1-dev
  )
  if ! (
    cd "${deb_root}"
    apt-get download "${build_packages[@]}"
    shopt -s nullglob
    for deb_file in ./*.deb; do
      dpkg-deb -x "${deb_file}" "${sysroot}"
    done
  ) >>"${build_log}" 2>&1; then
    tail -n 100 "${build_log}" >&2
    exit 1
  fi
fi

# Development packages contain linker symlinks while the corresponding
# versioned shared objects live in the worker image.  Complete only missing
# symlink targets locally; this keeps the build isolated and non-destructive.
if [[ -d "${sysroot_lib_dir}" ]]; then
  shopt -s nullglob
  for linker in "${sysroot_lib_dir}"/*.so; do
    link_target="$(readlink "${linker}" || true)"
    [[ -n "${link_target}" && ! -e "${linker}" ]] || continue
    for system_lib_dir in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
      candidate="${system_lib_dir}/${link_target}"
      if [[ -f "${candidate}" ]]; then
        ln -s "${candidate}" "${sysroot_lib_dir}/${link_target}"
        break
      fi
    done
  done
fi

for tool in apt-get curl dpkg-deb tar gcc make; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    printf 'Missing build tool: %s\n' "${tool}" >&2
    exit 1
  fi
done

if [[ ! -f "${local_ffi_root}/usr/include/x86_64-linux-gnu/ffi.h" ]]; then
  deb_root="${deps_root}/debs"
  mkdir -p "${deb_root}" "${local_ffi_root}"
  if ! (
    cd "${deb_root}"
    apt-get download libffi-dev
    deb_file="$(find . -maxdepth 1 -type f -name 'libffi-dev_*.deb' -print -quit)"
    test -n "${deb_file}"
    dpkg-deb -x "${deb_file}" "${local_ffi_root}"
  ) >>"${build_log}" 2>&1; then
    tail -n 100 "${build_log}" >&2
    exit 1
  fi
fi
ffi_lib_dir="${local_ffi_root}/usr/lib/x86_64-linux-gnu"
ffi_soname="$(readlink "${ffi_lib_dir}/libffi.so")"
if [[ ! -e "${ffi_lib_dir}/${ffi_soname}" ]]; then
  system_ffi="$(readlink -f /usr/lib/x86_64-linux-gnu/libffi.so.8)"
  if [[ ! -f "${system_ffi}" ]]; then
    printf 'Missing system libffi runtime: %s\n' "${system_ffi}" >&2
    exit 1
  fi
  ln -s "${system_ffi}" "${ffi_lib_dir}/${ffi_soname}"
fi
if [[ ! -f "${archive}" ]]; then
  curl --fail --location --retry 3 --retry-delay 2 \
    "https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz" \
    --output "${archive}"
fi
if [[ ! -d "${source_dir}" ]]; then
  if ! tar -xzf "${archive}" -C "${build_root}" >>"${build_log}" 2>&1; then
    tail -n 100 "${build_log}" >&2
    exit 1
  fi
fi

if [[ -f "${source_dir}/Makefile" ]]; then
  make -C "${source_dir}" distclean >>"${build_log}" 2>&1 || true
fi
export CPPFLAGS="-I${sysroot_include} -I${sysroot_include}/uuid -I${sysroot_multiarch_include} -I${local_ffi_root}/usr/include/x86_64-linux-gnu${CPPFLAGS:+ ${CPPFLAGS}}"
export LDFLAGS="-L${sysroot_lib_dir} -L${local_ffi_root}/usr/lib/x86_64-linux-gnu${LDFLAGS:+ ${LDFLAGS}}"
export PKG_CONFIG_PATH="${sysroot_lib_dir}/pkgconfig:${local_ffi_root}/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
if ! (
  cd "${source_dir}"
  ./configure --prefix="${build_root}" --with-ensurepip=install
) >>"${build_log}" 2>&1; then
  tail -n 100 "${build_log}" >&2
  exit 1
fi

if ! make -C "${source_dir}" -j "${PYTHON_BUILD_JOBS:-16}" >>"${build_log}" 2>&1; then
  tail -n 100 "${build_log}" >&2
  exit 1
fi
if ! make -C "${source_dir}" altinstall >>"${build_log}" 2>&1; then
  tail -n 100 "${build_log}" >&2
  exit 1
fi

"${python_bin}" --version
"${python_bin}" -c "import ${module_imports}; print('standard modules available')"
"${python_bin}" -m pip --version
