# OpenGL 3.3 Core GLSL Shaders for RLLS 16

STARFIELD_VS = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in float in_brightness;

uniform mat4 u_view;
uniform mat4 u_proj;

out float v_brightness;

void main() {
    v_brightness = in_brightness;
    // Remove translation from view matrix so stars remain infinitely far away
    mat4 view_rot = mat4(mat3(u_view));
    vec4 pos = u_proj * view_rot * vec4(in_position, 1.0);
    gl_Position = pos.xyww; // Force z = w to render at depth = 1.0 (furthest back)
    gl_PointSize = 1.5 + in_brightness * 1.5;
}
"""

STARFIELD_FS = """
#version 330 core
in float v_brightness;
out vec4 frag_color;

void main() {
    float alpha = clamp(v_brightness, 0.2, 1.0);
    vec3 star_color = mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.95, 0.85), fract(v_brightness * 3.7));
    frag_color = vec4(star_color * alpha, 1.0);
}
"""

SUN_VS = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_normal;
out vec3 v_world_pos;
out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = mat3(u_model) * in_normal;
    v_uv = in_uv;
    gl_Position = u_proj * u_view * world_pos;
}
"""

SUN_FS = """
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;
in vec2 v_uv;

uniform vec3 u_camera_pos;
uniform float u_time;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    float limb = clamp(dot(N, V), 0.0, 1.0);
    
    // Solar limb darkening and radiant core
    vec3 core_color = vec3(1.0, 0.96, 0.82);
    vec3 edge_color = vec3(1.0, 0.52, 0.12);
    vec3 corona_glow = vec3(1.0, 0.75, 0.25);
    
    vec3 color = mix(edge_color, core_color, pow(limb, 0.5));
    
    // Subtle granulation pattern
    float gran = sin(v_uv.x * 40.0 + u_time * 0.4) * cos(v_uv.y * 40.0 + u_time * 0.3) * 0.04;
    color += vec3(gran);
    
    // Outer corona fringe
    float rim = pow(1.0 - limb, 2.5) * 0.6;
    color += corona_glow * rim;

    frag_color = vec4(color * 1.3, 1.0);
}
"""

PLANET_VS = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_uv = in_uv;
    gl_Position = u_proj * u_view * world_pos;
}
"""

EARTH_FS = """
#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

uniform vec3 u_camera_pos;
uniform vec3 u_sun_pos;
uniform vec3 u_sun_dir;
uniform sampler2D u_tex_albedo;
uniform sampler2D u_tex_clouds;
uniform float u_cloud_offset;
uniform float u_show_clouds;
uniform float u_show_atmo;
uniform float u_solar_irradiance; // mutable environmental modifier
uniform int u_debug_mode;         // 0: Normal, 1: Solid, 2: Wireframe, 3: Height, 4: Biome, 5: Land, 6: Ocean, 7: Normals

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = length(u_sun_dir) > 0.01 ? normalize(u_sun_dir) : normalize(u_sun_pos - v_world_pos);
    if (dot(L, L) < 0.01) {
        L = normalize(vec3(1.0, 0.25, 0.65));
    }
    vec3 V = normalize(u_camera_pos - v_world_pos); // Direction to Camera
    vec3 H = normalize(L + V);

    if (u_debug_mode == 1) {
        // Solid Earth Debug Mode
        float diff = max(dot(N, L), 0.15);
        frag_color = vec4(vec3(0.12, 0.58, 0.88) * diff, 1.0);
        return;
    } else if (u_debug_mode == 7) {
        frag_color = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }

    float NdotL = dot(N, L);
    // Smooth day/night terminator
    float sun_light = smoothstep(-0.15, 0.15, NdotL);
    float ambient = 0.04;

    // Sample terrain albedo from canonical world texture
    vec4 albedo_sample = texture(u_tex_albedo, v_uv);
    vec3 surface_color = albedo_sample.rgb;
    float is_land = albedo_sample.a; // alpha stores land_mask (1 = land, 0 = ocean)

    if (u_debug_mode == 5) {
        frag_color = vec4(vec3(is_land > 0.5 ? 1.0 : 0.0), 1.0);
        return;
    } else if (u_debug_mode == 6) {
        frag_color = vec4(vec3(0.05, 0.35, 0.85) * (is_land < 0.5 ? 1.0 : 0.0), 1.0);
        return;
    }

    // Ocean specular reflection (water highlights on day side)
    float spec = 0.0;
    if (is_land < 0.5 && NdotL > 0.0) {
        spec = pow(max(dot(N, H), 0.0), 32.0) * 0.75 * sun_light;
    }

    // Clouds
    if (u_show_clouds > 0.5) {
        vec2 cloud_uv = vec2(v_uv.x + u_cloud_offset, v_uv.y);
        vec4 cloud_sample = texture(u_tex_clouds, cloud_uv);
        float cloud_cover = cloud_sample.r;
        surface_color = mix(surface_color, vec3(0.95, 0.98, 1.0), cloud_cover * 0.85);
    }

    // Lit surface color
    vec3 lit_color = surface_color * (ambient + (sun_light * 0.96 * u_solar_irradiance)) + vec3(spec);

    // Night side: faint scientific blue / city luminescence on land
    vec3 night_color = vec3(0.015, 0.025, 0.045);
    if (is_land > 0.5) {
        night_color += vec3(0.02, 0.035, 0.05);
    }

    vec3 final_color = mix(night_color, lit_color, sun_light);

    // Atmospheric scattering rim
    if (u_show_atmo > 0.5) {
        float rim = pow(1.0 - max(dot(N, V), 0.0), 3.5);
        vec3 atmo_color = vec3(0.22, 0.55, 0.95);
        // Atmosphere is more intense on the sunlit limb
        float atmo_sun = smoothstep(-0.2, 0.4, NdotL);
        final_color += atmo_color * rim * (0.2 + 0.8 * atmo_sun);
    }

    frag_color = vec4(final_color, 1.0);
}
"""

MOON_FS = """
#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

uniform vec3 u_sun_pos;
uniform vec3 u_sun_dir;
uniform sampler2D u_tex_lunar;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = length(u_sun_dir) > 0.01 ? normalize(u_sun_dir) : normalize(u_sun_pos - v_world_pos);
    if (dot(L, L) < 0.01) {
        L = normalize(vec3(1.0, 0.25, 0.65));
    }
    float NdotL = dot(N, L);
    float sun_light = smoothstep(-0.05, 0.08, NdotL);
    float ambient = 0.02;

    vec3 lunar_albedo = texture(u_tex_lunar, v_uv).rgb;
    vec3 color = lunar_albedo * (ambient + sun_light * 0.98);

    frag_color = vec4(color, 1.0);
}
"""

LINE_VS = """
#version 330 core
layout(location = 0) in vec3 in_position;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
}
"""

LINE_FS = """
#version 330 core
uniform vec4 u_line_color;
out vec4 frag_color;

void main() {
    frag_color = u_line_color;
}
"""

OVERLAY_VS = """
#version 330 core
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;

out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

OVERLAY_FS = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_ui_texture;
out vec4 frag_color;

void main() {
    vec4 col = texture(u_ui_texture, v_uv);
    frag_color = col;
}
"""

CUBESPHERE_TERRAIN_VS = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec2 in_scalars; // .x = elevation, .y = land_mask

uniform mat4 u_model; // Model matrix (or camera-relative model matrix)
uniform mat4 u_view;  // View matrix
uniform mat4 u_proj;  // Projection matrix

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out float v_elev;
out float v_land;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_uv = in_uv;
    v_elev = in_scalars.x;
    v_land = in_scalars.y;
    gl_Position = u_proj * u_view * world_pos;
}
"""

CUBESPHERE_TERRAIN_FS = """
#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in float v_elev;
in float v_land;

uniform vec3 u_camera_pos;         // Camera position in world space
uniform vec3 u_sun_pos;            // Sun position in world space
uniform vec3 u_sun_dir;            // Direction vector to Sun (astronomical)
uniform sampler2D u_tex_albedo;    // Canonical Earth albedo texture
uniform sampler2D u_tex_clouds;    // Cloud texture
uniform float u_cloud_offset;      // Cloud rotation offset
uniform float u_show_clouds;       // Cloud toggle
uniform float u_show_atmo;         // Atmosphere toggle
uniform float u_solar_irradiance;  // Solar irradiance multiplier
uniform int u_debug_mode;          // 0: Normal, 1: Solid, 2: Wireframe, 3: Height, 4: Biome, 5: Land Mask, 6: Ocean Mask, 7: Normals, 8: LOD, 9: Tile Bounds
uniform float u_lod_level;         // Node LOD level for debug tint

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = length(u_sun_dir) > 0.01 ? normalize(u_sun_dir) : normalize(u_sun_pos - v_world_pos);
    if (dot(L, L) < 0.01) {
        L = normalize(vec3(1.0, 0.25, 0.65));
    }

    // -------------------------------------------------------------
    // VISUAL DEBUG MODES (Point 44)
    // -------------------------------------------------------------
    if (u_debug_mode == 1) {
        // 1. Earth Solid (clean cyan-blue diagnostic surface)
        float diff = max(dot(N, L), 0.15);
        frag_color = vec4(vec3(0.12, 0.58, 0.88) * diff, 1.0);
        return;
    }
    else if (u_debug_mode == 3) {
        // 3. Heightmap / Elevation gradient
        float h = clamp((v_elev - 0.1) / 0.85, 0.0, 1.0);
        frag_color = vec4(vec3(h), 1.0);
        return;
    }
    else if (u_debug_mode == 4) {
        // 4. Biome coloration from raw albedo texture
        vec3 biome_col = texture(u_tex_albedo, v_uv).rgb;
        frag_color = vec4(biome_col, 1.0);
        return;
    }
    else if (u_debug_mode == 5) {
        // 5. Land Mask (white = land, black = ocean)
        float m = v_land > 0.5 ? 1.0 : 0.0;
        frag_color = vec4(vec3(m), 1.0);
        return;
    }
    else if (u_debug_mode == 6) {
        // 6. Ocean Mask (deep ocean blue, black = land)
        float o = v_land < 0.5 ? 1.0 : 0.0;
        frag_color = vec4(vec3(0.05, 0.35, 0.85) * o, 1.0);
        return;
    }
    else if (u_debug_mode == 7) {
        // 7. Surface Normals
        frag_color = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }
    else if (u_debug_mode == 8) {
        // 8. LOD level coloring
        vec3 lod_colors[7] = vec3[](
            vec3(0.9, 0.2, 0.2), // LOD 0: Red
            vec3(0.9, 0.6, 0.1), // LOD 1: Orange
            vec3(0.9, 0.9, 0.2), // LOD 2: Yellow
            vec3(0.2, 0.9, 0.3), // LOD 3: Green
            vec3(0.2, 0.8, 0.9), // LOD 4: Cyan
            vec3(0.3, 0.3, 0.9), // LOD 5: Blue
            vec3(0.8, 0.2, 0.9)  // LOD 6: Magenta
        );
        int idx = clamp(int(u_lod_level), 0, 6);
        float diff = max(dot(N, L), 0.25);
        frag_color = vec4(lod_colors[idx] * diff, 1.0);
        return;
    }
    else if (u_debug_mode == 9) {
        // 9. Tile Bounds (grid borders)
        vec2 grid = abs(fract(v_uv * 32.0 - 0.5) - 0.5) / fwidth(v_uv * 32.0);
        float line = min(grid.x, grid.y);
        float c = 1.0 - min(line, 1.0);
        vec3 base = texture(u_tex_albedo, v_uv).rgb * 0.6;
        frag_color = vec4(mix(base, vec3(1.0, 0.9, 0.1), c), 1.0);
        return;
    }

    // -------------------------------------------------------------
    // FULL CANONICAL EARTH SHADING PIPELINE (Points 18, 19, 20)
    // -------------------------------------------------------------
    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float NdotL = dot(N, L);
    // Smooth day/night terminator (Point 19)
    float daylight = smoothstep(-0.08, 0.12, NdotL);
    float night_ambient = 0.035;

    // Sample Canonical Earth texture
    vec4 albedo_sample = texture(u_tex_albedo, v_uv);
    vec3 surface_color = albedo_sample.rgb;

    // -------------------------------------------------------------
    // INCREASING SPATIAL DETAIL AT DEEP ZOOM
    // -------------------------------------------------------------
    float cam_dist = length(u_camera_pos - v_world_pos);
    float detail_factor = clamp(1.0 - (cam_dist - 5.0) / 1.5, 0.0, 1.0);

    if (detail_factor > 0.01) {
        // Slope-Aware Shading: Steeper slopes blend toward rocky cliff face
        float slope = 1.0 - clamp(dot(N, normalize(v_world_pos)), 0.0, 1.0);
        if (v_land > 0.5 && slope > 0.15) {
            float cliff_blend = smoothstep(0.15, 0.45, slope) * detail_factor;
            vec3 rock_color = vec3(0.38, 0.35, 0.32);
            surface_color = mix(surface_color, rock_color, cliff_blend * 0.75);
        }

        // Procedural Micro-grain on land surface
        if (v_land > 0.5) {
            float grain = sin(v_world_pos.x * 650.0) * cos(v_world_pos.z * 650.0) * 0.035;
            surface_color += vec3(grain * detail_factor);
        } else {
            // Procedural Micro-ripples on ocean surface
            float wave = sin(v_world_pos.x * 400.0 + u_cloud_offset * 12.0) * cos(v_world_pos.z * 400.0) * 0.025;
            surface_color += vec3(0.02, 0.05, 0.09) * wave * detail_factor;
        }
    }

    // Ocean specular reflection (water glint on daylight side)
    if (v_land < 0.5) {
        float NdotH = max(dot(N, H), 0.0);
        float spec_exp = mix(48.0, 192.0, detail_factor);
        float spec = pow(NdotH, spec_exp) * max(NdotL, 0.0);
        surface_color += vec3(0.85, 0.92, 1.0) * (spec * 0.70 * u_solar_irradiance);
    }

    // Dynamic Cloud Layer
    if (u_show_clouds > 0.5) {
        vec2 cloud_uv = vec2(fract(v_uv.x + u_cloud_offset), v_uv.y);
        vec4 cloud_sample = texture(u_tex_clouds, cloud_uv);
        float cloud_alpha = cloud_sample.r;
        vec3 cloud_lit = vec3(0.96, 0.98, 1.0) * (max(NdotL, 0.0) * 0.92 + 0.08);
        surface_color = mix(surface_color, cloud_lit, cloud_alpha * 0.78);
    }

    // Direct daylight + night ambient
    vec3 lit_color = surface_color * (daylight * u_solar_irradiance + night_ambient);

    // Night luminescence on continents
    if (v_land > 0.5 && daylight < 0.2) {
        vec3 night_glow = vec3(0.012, 0.022, 0.038) * (1.0 - daylight * 5.0);
        lit_color += night_glow;
    }

    // Atmospheric limb scattering (Rayleigh rim glow, Point 20)
    if (u_show_atmo > 0.5) {
        float NdotV = clamp(dot(N, V), 0.0, 1.0);
        float rim = pow(1.0 - NdotV, 3.2);
        vec3 atmo_color = vec3(0.25, 0.55, 1.0) * (max(NdotL, 0.0) * 0.85 + 0.15);
        lit_color += atmo_color * rim * 0.65;
    }

    frag_color = vec4(lit_color, 1.0);
}
"""

VEGETATION_VS = """
#version 330 core
layout(location = 0) in vec3 in_position;    // Local tree geometry (trunk + canopy)
layout(location = 1) in vec3 in_normal;      // Local vertex normal
layout(location = 2) in float in_part;       // 0.0 = trunk, 1.0 = foliage canopy

// Per-instance attributes (hardware instanced)
layout(location = 3) in vec3 in_inst_pos;    // World position on terrain surface
layout(location = 4) in float in_inst_scale; // Tree scale
layout(location = 5) in float in_inst_type;  // 0: broadleaf, 1: conifer, 2: rainforest
layout(location = 6) in vec3 in_inst_normal; // Surface normal

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_sun_pos;
uniform vec3 u_sun_dir;
uniform float u_time;

out vec3 v_world_pos;
out vec3 v_normal;
out float v_part;
out float v_type;

void main() {
    vec3 N = normalize(in_inst_normal);
    // Build local orthonormal tangent frame aligned with planet surface normal
    vec3 up_ref = abs(N.y) > 0.85 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 T = normalize(cross(up_ref, N));
    vec3 B = cross(N, T);

    // Subtle wind sway for foliage canopy
    vec3 local_pos = in_position;
    if (in_part > 0.5) {
        float sway = sin(u_time * 2.5 + in_inst_pos.x * 20.0 + in_inst_pos.z * 20.0) * 0.08;
        local_pos.x += sway * local_pos.y;
        local_pos.z += sway * 0.7 * local_pos.y;
    }

    // Orient local tree mesh along surface normal
    vec3 oriented_pos = (T * local_pos.x + N * local_pos.y + B * local_pos.z) * in_inst_scale;
    vec3 world_pos = in_inst_pos + oriented_pos;

    v_world_pos = world_pos;
    v_normal = normalize(T * in_normal.x + N * in_normal.y + B * in_normal.z);
    v_part = in_part;
    v_type = in_inst_type;

    gl_Position = u_proj * u_view * vec4(world_pos, 1.0);
}
"""

VEGETATION_FS = """
#version 330 core
in vec3 v_world_pos;
in vec3 v_normal;
in float v_part;
in float v_type;

uniform vec3 u_camera_pos;
uniform vec3 u_sun_pos;
uniform vec3 u_sun_dir;
uniform float u_solar_irradiance;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = length(u_sun_dir) > 0.01 ? normalize(u_sun_dir) : normalize(u_sun_pos - v_world_pos);
    if (dot(L, L) < 0.01) {
        L = normalize(vec3(1.0, 0.25, 0.65));
    }

    float NdotL = max(dot(N, L), 0.0);
    float daylight = smoothstep(-0.05, 0.15, dot(N, L));
    float ambient = 0.15;

    vec3 base_color;
    if (v_part < 0.5) {
        // Tree trunk: woody bark
        base_color = vec3(0.32, 0.21, 0.13);
    } else {
        // Foliage canopy colored by biome species
        if (v_type < 0.5) {
            // Temperate forest oak/broadleaf
            base_color = vec3(0.18, 0.52, 0.20);
        } else if (v_type < 1.5) {
            // Taiga / Boreal spruce conifer
            base_color = vec3(0.10, 0.38, 0.22);
        } else {
            // Tropical rainforest canopy
            base_color = vec3(0.08, 0.58, 0.25);
        }
    }

    vec3 final_color = base_color * (daylight * max(0.2, u_solar_irradiance) + ambient);
    frag_color = vec4(final_color, 1.0);
}
"""
