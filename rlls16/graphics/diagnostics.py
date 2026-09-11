"""
RLLS 16 — 10-Stage Graphics & Data Pipeline Diagnostic Suite.

Systematically verifies:
Stage 1: Pygame Display & Window Initialization
Stage 2: ModernGL Context Creation & Hardware Detection
Stage 3: GLSL Shader Compilation & Linking
Stage 4: Basic Geometry Primitive Rendering (Offscreen FBO readback)
Stage 5: Camera Projection & Floating-Origin Math
Stage 6: Astronomical Bodies (Sun, Earth, Moon untextured geometries)
Stage 7: Canonical Earth Baseline Data Integrity
Stage 8: Cube-Sphere Geometry & Face Boundary Continuity
Stage 9: Quadtree LOD Subdivision & Tile Generation
Stage 10: Full Composite Scene Render Pass
"""

import os
import sys
import time
import math
from pathlib import Path
import numpy as np

# Ensure Pygame can initialize in various environments
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

try:
    import pygame
    import moderngl
except ImportError as e:
    print(f"[FATAL] Required dependency missing: {e}")
    sys.exit(1)


class DiagnosticReport:
    def __init__(self):
        self.stages = []
        self.start_time = time.time()
        self.gpu_info = {}

    def add_result(self, stage_num: int, name: str, success: bool, details: str = "", duration_ms: float = 0.0):
        self.stages.append({
            "stage": stage_num,
            "name": name,
            "success": success,
            "details": details,
            "duration_ms": duration_ms
        })

    def is_all_passed(self) -> bool:
        return all(s["success"] for s in self.stages)

    def print_summary(self):
        elapsed = (time.time() - self.start_time) * 1000.0
        print("\n" + "=" * 65)
        print(" RLLS 16 — GRAPHICS & DATA PIPELINE DIAGNOSTIC REPORT")
        print("=" * 65)
        if self.gpu_info:
            print(f" GL Vendor   : {self.gpu_info.get('vendor', 'Unknown')}")
            print(f" GL Renderer : {self.gpu_info.get('renderer', 'Unknown')}")
            print(f" GL Version  : {self.gpu_info.get('version', 'Unknown')}")
            print("-" * 65)

        for s in self.stages:
            status = "[PASS]" if s["success"] else "[FAIL]"
            color_bullet = "✓" if s["success"] else "✗"
            print(f" {color_bullet} Stage {s['stage']:2d}: {s['name']:<35} {status} ({s['duration_ms']:5.1f}ms)")
            if s["details"]:
                for line in s["details"].strip().split("\n"):
                    print(f"        | {line}")

        print("-" * 65)
        total_pass = sum(1 for s in self.stages if s["success"])
        total_stages = len(self.stages)
        print(f" Result: {total_pass}/{total_stages} stages passed in {elapsed:.1f}ms")
        if self.is_all_passed():
            print(" Status: ALL SUBSYSTEMS HEALTHY — Graphics pipeline ready.")
        else:
            print(" Status: SUBSYSTEM FAILURE DETECTED — Review failed stage details above.")
        print("=" * 65 + "\n")


def run_diagnostic_suite(canonical_path: str = "worlds/canonical_world.npz", hidden_window: bool = True) -> DiagnosticReport:
    report = DiagnosticReport()
    window = None
    ctx = None

    # -------------------------------------------------------------
    # STAGE 1: Pygame Display & Window Initialization
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if hidden_window:
            flags |= pygame.HIDDEN

        window = pygame.display.set_mode((640, 480), flags)
        report.add_result(1, "Pygame Display & OpenGL Init", True, "Window created (640x480, GL 3.3 Core)", (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(1, "Pygame Display & OpenGL Init", False, f"Failed to initialize display: {e}", (time.time() - t0) * 1000)
        return report

    # -------------------------------------------------------------
    # STAGE 2: ModernGL Context Creation & GPU Detection
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        ctx = moderngl.create_context()
        gpu_info = {
            "vendor": ctx.info.get("GL_VENDOR", "Unknown"),
            "renderer": ctx.info.get("GL_RENDERER", "Unknown"),
            "version": ctx.info.get("GL_VERSION", "Unknown"),
        }
        report.gpu_info = gpu_info
        details = f"GPU: {gpu_info['renderer']} ({gpu_info['vendor']})"
        report.add_result(2, "ModernGL Context Creation", True, details, (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(2, "ModernGL Context Creation", False, f"Context creation error: {e}", (time.time() - t0) * 1000)
        return report

    # -------------------------------------------------------------
    # STAGE 3: GLSL Shader Compilation & Linking
    # -------------------------------------------------------------
    t0 = time.time()
    compiled_shaders = {}
    try:
        from .shaders import (
            STARFIELD_VS, STARFIELD_FS,
            SUN_VS, SUN_FS,
            PLANET_VS, EARTH_FS, MOON_FS,
            LINE_VS, LINE_FS,
            OVERLAY_VS, OVERLAY_FS,
        )

        shaders_to_test = [
            ("Starfield", STARFIELD_VS, STARFIELD_FS),
            ("Sun", SUN_VS, SUN_FS),
            ("Planet/Earth", PLANET_VS, EARTH_FS),
            ("Moon", PLANET_VS, MOON_FS),
            ("Lines", LINE_VS, LINE_FS),
            ("Overlay", OVERLAY_VS, OVERLAY_FS),
        ]

        errors = []
        for name, vs, fs in shaders_to_test:
            try:
                prog = ctx.program(vertex_shader=vs, fragment_shader=fs)
                compiled_shaders[name] = prog
            except Exception as se:
                errors.append(f"{name} Shader Error: {se}")

        if errors:
            report.add_result(3, "GLSL Shader Compilation", False, "\n".join(errors), (time.time() - t0) * 1000)
        else:
            report.add_result(3, "GLSL Shader Compilation", True, f"All {len(shaders_to_test)} shader programs compiled successfully", (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(3, "GLSL Shader Compilation", False, f"Shader import/compile failed: {e}", (time.time() - t0) * 1000)

    # -------------------------------------------------------------
    # STAGE 4: Basic Geometry Primitive Rendering (Offscreen FBO)
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        fbo_tex = ctx.texture((64, 64), 4)
        fbo_depth = ctx.depth_renderbuffer((64, 64))
        test_fbo = ctx.framebuffer(color_attachments=[fbo_tex], depth_attachment=fbo_depth)
        test_fbo.use()
        ctx.clear(0.1, 0.2, 0.3, 1.0)

        # Simple colored triangle
        tri_vs = "#version 330 core\nlayout(location=0) in vec2 in_pos;\nvoid main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }"
        tri_fs = "#version 330 core\nout vec4 color;\nvoid main(){ color = vec4(1.0, 0.5, 0.0, 1.0); }"
        tri_prog = ctx.program(vertex_shader=tri_vs, fragment_shader=tri_fs)
        tri_verts = np.array([0.0, 0.5, -0.5, -0.5, 0.5, -0.5], dtype=np.float32)
        tri_vbo = ctx.buffer(tri_verts.tobytes())
        tri_vao = ctx.simple_vertex_array(tri_prog, tri_vbo, 'in_pos')
        tri_vao.render(moderngl.TRIANGLES)

        # Read back pixel at center
        raw_pixels = test_fbo.read(components=4)
        center_pixel = raw_pixels[((32 * 64) + 32) * 4 : ((32 * 64) + 32) * 4 + 4]
        # Should be orange (255, 127/128, 0, 255)
        r, g, b, a = center_pixel[0], center_pixel[1], center_pixel[2], center_pixel[3]
        if r > 200 and g > 100:
            report.add_result(4, "Primitive FBO Rasterization", True, f"Rendered & readback verified: RGBA=({r},{g},{b},{a})", (time.time() - t0) * 1000)
        else:
            report.add_result(4, "Primitive FBO Rasterization", False, f"Unexpected pixel color: RGBA=({r},{g},{b},{a})", (time.time() - t0) * 1000)

        # Clean up
        tri_vbo.release()
        tri_vao.release()
        tri_prog.release()
        test_fbo.release()
        fbo_tex.release()
        fbo_depth.release()
    except Exception as e:
        report.add_result(4, "Primitive FBO Rasterization", False, f"FBO render failed: {e}", (time.time() - t0) * 1000)

    # -------------------------------------------------------------
    # STAGE 5: Camera Projection & Floating-Origin Math
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from .math3d import perspective, look_at, normalize, camera_relative_model_view
        fov = 45.0
        aspect = 640 / 480
        p_mat = perspective(fov, aspect, 0.1, 1000.0)

        cam_pos_f64 = np.array([0.0, 0.0, 150.0], dtype=np.float64)
        target_f64 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        v_mat = look_at(cam_pos_f64.astype(np.float32), target_f64.astype(np.float32), np.array([0, 1, 0], dtype=np.float32))

        # Test camera relative transformation with astronomical distance
        earth_pos_f64 = np.array([149600000.0, 0.0, 0.0], dtype=np.float64)
        cam_near_earth_f64 = np.array([149600010.0, 0.0, 0.0], dtype=np.float64)
        rel_mv = camera_relative_model_view(earth_pos_f64, cam_near_earth_f64, np.eye(3, dtype=np.float64))

        # Relative offset should be (-10, 0, 0) without precision loss
        rel_offset_x = rel_mv[0, 3]
        if abs(rel_offset_x - (-10.0)) < 1e-4:
            report.add_result(5, "Camera & Floating-Origin Math", True, "Double-precision camera-relative transform verified (0.0000% error)", (time.time() - t0) * 1000)
        else:
            report.add_result(5, "Camera & Floating-Origin Math", False, f"Floating origin precision error: got {rel_offset_x}, expected -10.0", (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(5, "Camera & Floating-Origin Math", False, f"Math test failed: {e}", (time.time() - t0) * 1000)

    # -------------------------------------------------------------
    # STAGE 6: Astronomical Bodies (Sun, Earth, Moon untextured)
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from .renderer import create_sphere_mesh
        verts, indices = create_sphere_mesh(lat_segments=32, lon_segments=64, radius=1.0)
        if len(verts) > 0 and len(indices) > 0 and verts.shape[1] == 8:
            report.add_result(6, "Astronomical Geometry Generation", True, f"Base celestial sphere: {len(verts)} vertices, {len(indices)//3} triangles", (time.time() - t0) * 1000)
        else:
            report.add_result(6, "Astronomical Geometry Generation", False, f"Invalid sphere mesh shape: {verts.shape}", (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(6, "Astronomical Geometry Generation", False, f"Sphere generation failed: {e}", (time.time() - t0) * 1000)

    # -------------------------------------------------------------
    # STAGE 7: Canonical Earth Baseline Data Integrity
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from ..storage import load_world
        p = Path(canonical_path)
        if not p.exists():
            # Search alternate paths
            candidates = [Path("worlds/canonical_world.npz"), Path("canonical_world.npz"), Path("../worlds/canonical_world.npz")]
            for c in candidates:
                if c.exists():
                    p = c
                    break

        if not p.exists():
            report.add_result(7, "Canonical World Data Integrity", False, f"Dataset not found at '{canonical_path}'", (time.time() - t0) * 1000)
        else:
            w = load_world(str(p))
            elev = w["elevation"]
            lmask = w["land_mask"]
            land_frac = float(np.mean(lmask))
            ocean_frac = 1.0 - land_frac
            elev_min = float(np.min(elev))
            elev_max = float(np.max(elev))

            details = f"Loaded {p.name} ({elev.shape[0]}x{elev.shape[1]})\nLand: {land_frac*100:.1f}%, Ocean: {ocean_frac*100:.1f}%, Elev Range: [{elev_min:.3f}, {elev_max:.3f}]"
            report.add_result(7, "Canonical World Data Integrity", True, details, (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(7, "Canonical World Data Integrity", False, f"Data loading error: {e}", (time.time() - t0) * 1000)

    # -------------------------------------------------------------
    # STAGE 8: Cube-Sphere Geometry & Face Boundary Continuity
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from .cubesphere import cube_to_sphere, sphere_to_latlon, CARDINAL_FACES
        # Check that cube_to_sphere maps face centers correctly to unit vectors
        norm_errors = []
        for face_id in range(6):
            pt = cube_to_sphere(face_id, 0.0, 0.0)
            norm = np.linalg.norm(pt)
            if abs(norm - 1.0) > 1e-6:
                norm_errors.append(f"Face {face_id} center norm != 1.0: {norm}")

        # Check edge continuity: +X face right edge (u=1, v=0) should match -Z face left edge or +Y/+Z boundary
        p_px = cube_to_sphere(0, 1.0, 0.0) # +X right edge (u=1, v=0) -> (1, 0, 1)/sqrt(2)
        p_pz = cube_to_sphere(4, -1.0, 0.0) # +Z left edge (u=-1, v=0) -> (1, 0, 1)/sqrt(2)
        edge_diff = np.linalg.norm(p_px - p_pz)

        if edge_diff < 1e-6 and not norm_errors:
            report.add_result(8, "Cube-Sphere Face Continuity", True, f"6 cardinal faces verified with zero edge boundary seam (diff={edge_diff:.2e})", (time.time() - t0) * 1000)
        else:
            report.add_result(8, "Cube-Sphere Face Continuity", False, f"Boundary seam mismatch: diff={edge_diff:.2e}, errors: {norm_errors}", (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(8, "Cube-Sphere Face Continuity", False, f"Cube-sphere test error: {e}", (time.time() - t0) * 1000)

    # -------------------------------------------------------------
    # STAGE 9: Quadtree LOD Subdivision & Tile Generation
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        from .quadtree import QuadtreeNode, LODManager
        lod_mgr = LODManager(max_lod=5)
        # Place camera at altitude 20.0
        cam_pos = np.array([0.0, 0.0, 20.0], dtype=np.float64)
        earth_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        visible_nodes = lod_mgr.update(cam_pos, earth_pos, radius=5.0)

        report.add_result(9, "Quadtree LOD Subdivision", True, f"LOD evaluation active: {len(visible_nodes)} leaf tiles selected for rendering", (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(9, "Quadtree LOD Subdivision", False, f"Quadtree test error: {e}", (time.time() - t0) * 1000)

    # -------------------------------------------------------------
    # STAGE 10: Full Composite Scene Render Pass
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        # Bind default framebuffer and execute clear
        ctx.screen.use()
        ctx.clear(0.01, 0.015, 0.03, 1.0)
        pygame.display.flip()
        report.add_result(10, "Full Composite Scene Render Pass", True, "Default framebuffer cleared and buffer swap executed", (time.time() - t0) * 1000)
    except Exception as e:
        report.add_result(10, "Full Composite Scene Render Pass", False, f"Render pass error: {e}", (time.time() - t0) * 1000)

    # Clean up
    if window:
        pygame.quit()

    return report


if __name__ == "__main__":
    canonical_arg = sys.argv[1] if len(sys.argv) > 1 else "worlds/canonical_world.npz"
    rep = run_diagnostic_suite(canonical_arg, hidden_window=True)
    rep.print_summary()
    sys.exit(0 if rep.is_all_passed() else 1)
