"""
RLLS 16 — High-Performance GPU Configuration & Hardware Optimization.

Provides automated discrete / external GPU optimization:
1. Enforces Windows High-Performance GPU Preference (DirectX / WGL / Optimus)
2. Exports driver hints for NVIDIA Optimus and AMD PowerXpress
3. Configures hardware 4x MSAA (Multisample Anti-Aliasing) and 16x Anisotropic filtering
4. Expands GPU mesh cache pools for dedicated VRAM
5. Identifies and reports active GPU hardware
"""

import os
import sys
import platform
import ctypes


def apply_gpu_environment_hints():
    """Set process environment variables before graphics context creation."""
    os.environ["SHIM_MCCOMPAT"] = "0x800000001"
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["GPU_FORCE_64BIT_PTR"] = "1"
    os.environ["GPU_MAX_HEAP_SIZE"] = "100"
    os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"

    if platform.system() == "Windows":
        _configure_windows_gpu_preference()


def _configure_windows_gpu_preference():
    """Ensure current python executable has High-Performance GPU assigned in Windows registry."""
    try:
        import winreg
        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            # Add both current virtual environment and base python executables
            exes = set()
            if sys.executable:
                exes.add(os.path.abspath(sys.executable))
            base_exe = getattr(sys, "_base_executable", None)
            if base_exe:
                exes.add(os.path.abspath(base_exe))

            for exe in exes:
                try:
                    curr_val, _ = winreg.QueryValueEx(key, exe)
                    if "GpuPreference=2;" not in curr_val:
                        winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
                except FileNotFoundError:
                    winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
    except Exception:
        pass


def configure_pygame_gl_attributes():
    """Configure OpenGL attributes for high visual fidelity on dedicated GPUs."""
    import pygame
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

    # Enable 4x Hardware Multisample Anti-Aliasing (MSAA)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)


def get_gpu_hardware_info(ctx) -> dict:
    """Extract GPU vendor, renderer, and driver version from ModernGL context."""
    vendor = ctx.info.get("GL_VENDOR", "Unknown")
    renderer = ctx.info.get("GL_RENDERER", "Unknown")
    version = ctx.info.get("GL_VERSION", "Unknown")
    is_dedicated = any(kw in renderer.upper() or kw in vendor.upper() for kw in ["NVIDIA", "GEFORCE", "RTX", "GTX", "RADEON", "AMD", "DISCRETE", "ARC"])

    return {
        "vendor": vendor,
        "renderer": renderer,
        "version": version,
        "is_dedicated": is_dedicated,
        "max_anisotropy": ctx.info.get("GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT", 1.0) if hasattr(ctx, "info") else 1.0,
    }


def optimize_texture(tex, max_anisotropy: float = 16.0):
    """Apply high-quality anisotropic filtering to an OpenGL texture if supported."""
    try:
        if hasattr(tex, "anisotropy"):
            tex.anisotropy = min(16.0, max(1.0, float(max_anisotropy)))
    except Exception:
        pass
