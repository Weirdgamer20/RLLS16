import math
import time
from pathlib import Path
import pygame
from .theme import (
    PANEL_BG, PANEL_BORDER, PANEL_BORDER_FOCUS,
    ACCENT_CYAN, ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER,
    TEXT_PRIMARY, TEXT_MUTED, TEXT_DIM,
    BUTTON_BG, BUTTON_HOVER, BUTTON_ACTIVE, BUTTON_BORDER,
    get_fonts, draw_panel, draw_button
)

# Canonical 6 initial environments (Point 36)
INITIAL_ENVIRONMENTS = [
    "Coastal",
    "Forest",
    "Savannah",
    "Desert",
    "Mountain",
    "Tundra",
]

RL_GOALS = ["Survive", "Growth", "Exploration", "Culture"]


class HomeScreen:
    """
    Landing screen, 200ms animated creation modal, real multi-stage loading checklist,
    and world state loader.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fonts = get_fonts()

        # Point 39: Enable key-repeat for responsive text editing and Backspace
        pygame.key.set_repeat(400, 35)

        # State: 'MENU', 'MODAL_CREATE', 'MODAL_LOAD', 'LOADING'
        self.mode = "MENU"
        self.modal_open_time = 0.0

        # Create World Modal fields
        self.input_world_name = "World - 001"
        self.input_population = "1000"
        self.selected_env_idx = 0  # Coastal
        self.selected_rl_goal_idx = 0  # Survive
        self.rl_learning_enabled = True
        self.active_input = None   # 'name' or 'pop'

        # Loading progress state (Point 38)
        self.loading_start_time = 0.0
        self.loading_payload = None
        self.loading_stages = [
            "Loading canonical Earth",
            "Loading atmosphere",
            "Initializing Sun/Earth/Moon",
            "Initializing population",
            "Building terrain cache",
            "Preparing renderer",
            "Entering world",
        ]

        # Load World state
        self.saved_worlds = []
        self.selected_world_idx = 0

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height

    def open_create_modal(self):
        self.mode = "MODAL_CREATE"
        self.active_input = None
        self.modal_open_time = time.time()

    def open_load_modal(self):
        self.mode = "MODAL_LOAD"
        self.modal_open_time = time.time()
        self._refresh_saved_worlds()

    def _refresh_saved_worlds(self):
        self.saved_worlds = []
        sim_dir = Path("simulations")
        if sim_dir.exists():
            for p in sim_dir.iterdir():
                if p.is_dir() and (p / "world_state.json").exists():
                    self.saved_worlds.append(p.name.replace("_", " "))
        self.selected_world_idx = 0

    def handle_event(self, event) -> dict | None:
        """
        Handle UI input events.
        Returns an action dictionary if a button was triggered (e.g. {'action': 'CREATE', ...}).
        """
        if self.mode == "MENU":
            return self._handle_menu_event(event)
        elif self.mode == "MODAL_CREATE":
            return self._handle_create_event(event)
        elif self.mode == "MODAL_LOAD":
            return self._handle_load_event(event)
        elif self.mode == "LOADING":
            return None
        return None

    def _get_menu_rects(self):
        cx = self.width // 2
        cy = self.height // 2
        btn_w = 260
        btn_h = 38
        start_y = cy - 30
        rect_create = pygame.Rect(cx - btn_w // 2, start_y, btn_w, btn_h)
        rect_load = pygame.Rect(cx - btn_w // 2, start_y + 48, btn_w, btn_h)
        rect_exit = pygame.Rect(cx - btn_w // 2, start_y + 96, btn_w, btn_h)
        return rect_create, rect_load, rect_exit

    def _handle_menu_event(self, event) -> dict | None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            rect_create, rect_load, rect_exit = self._get_menu_rects()

            if rect_create.collidepoint(mx, my):
                self.open_create_modal()
            elif rect_load.collidepoint(mx, my):
                self.open_load_modal()
            elif rect_exit.collidepoint(mx, my):
                return {"action": "EXIT"}
        return None

    def _handle_create_event(self, event) -> dict | None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.mode = "MENU"
            elif event.key == pygame.K_TAB:
                self.active_input = "pop" if self.active_input == "name" else "name"
            elif self.active_input == "name":
                # Point 39: Text input & backspace handling
                if event.key == pygame.K_BACKSPACE:
                    self.input_world_name = self.input_world_name[:-1]
                elif event.key == pygame.K_DELETE:
                    self.input_world_name = ""
                elif len(self.input_world_name) < 24 and event.unicode.isprintable():
                    self.input_world_name += event.unicode
            elif self.active_input == "pop":
                if event.key == pygame.K_BACKSPACE:
                    self.input_population = self.input_population[:-1]
                elif event.key == pygame.K_DELETE:
                    self.input_population = ""
                elif len(self.input_population) < 9 and event.unicode.isdigit():
                    self.input_population += event.unicode

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            mw, mh = 520, 560
            cx, cy = self.width // 2, self.height // 2
            rx, ry = cx - mw // 2, cy - mh // 2

            # Input click checks
            name_rect = pygame.Rect(rx + 30, ry + 95, mw - 60, 28)
            pop_rect = pygame.Rect(rx + 30, ry + 152, mw - 60, 28)

            if name_rect.collidepoint(mx, my):
                self.active_input = "name"
            elif pop_rect.collidepoint(mx, my):
                self.active_input = "pop"
            else:
                self.active_input = None

            # Initial Environment radio buttons (Point 36)
            env_start_y = ry + 215
            for i, env_name in enumerate(INITIAL_ENVIRONMENTS):
                col = i % 2
                row = i // 2
                bx = rx + 30 + col * 235
                by = env_start_y + row * 30
                b_rect = pygame.Rect(bx, by, 220, 26)
                if b_rect.collidepoint(mx, my):
                    self.selected_env_idx = i

            # RL Goal selector (Point 30, 31)
            rl_btn_y = ry + 340
            for i, goal in enumerate(RL_GOALS):
                gx = rx + 30 + i * 115
                g_rect = pygame.Rect(gx, rl_btn_y, 108, 26)
                if g_rect.collidepoint(mx, my):
                    self.selected_rl_goal_idx = i

            # Action buttons
            btn_y = ry + mh - 55
            create_rect = pygame.Rect(rx + 30, btn_y, 220, 36)
            cancel_rect = pygame.Rect(rx + 270, btn_y, 220, 36)

            if create_rect.collidepoint(mx, my):
                pop_val = int(self.input_population) if self.input_population.isdigit() else 1000
                env_val = INITIAL_ENVIRONMENTS[self.selected_env_idx]
                goal_val = RL_GOALS[self.selected_rl_goal_idx]
                # Start multi-stage loading progress (Point 38)
                self.mode = "LOADING"
                self.loading_start_time = time.time()
                self.loading_payload = {
                    "action": "CREATE",
                    "world_name": self.input_world_name.strip() or "World - 001",
                    "human_population": pop_val,
                    "environment": env_val,
                    "rl_config": {
                        "goal": goal_val,
                        "learning_enabled": self.rl_learning_enabled,
                    }
                }
                return None

            elif cancel_rect.collidepoint(mx, my):
                self.mode = "MENU"

        return None

    def _handle_load_event(self, event) -> dict | None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.mode = "MENU"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            mw, mh = 480, 420
            cx, cy = self.width // 2, self.height // 2
            rx, ry = cx - mw // 2, cy - mh // 2

            list_y = ry + 75
            for i, name in enumerate(self.saved_worlds):
                item_rect = pygame.Rect(rx + 30, list_y + i * 36, mw - 60, 30)
                if item_rect.collidepoint(mx, my):
                    self.selected_world_idx = i

            btn_y = ry + mh - 55
            load_rect = pygame.Rect(rx + 30, btn_y, 195, 36)
            cancel_rect = pygame.Rect(rx + 255, btn_y, 195, 36)

            if load_rect.collidepoint(mx, my) and self.saved_worlds:
                chosen = self.saved_worlds[self.selected_world_idx]
                return {
                    "action": "LOAD",
                    "path": str(Path("simulations") / chosen.replace(" ", "_") / "world_state.json")
                }
            elif cancel_rect.collidepoint(mx, my):
                self.mode = "MENU"

        return None

    def check_loading_finished(self) -> dict | None:
        """Called by main app loop to transition when loading checklist finishes."""
        if self.mode == "LOADING":
            elapsed = time.time() - self.loading_start_time
            if elapsed >= len(self.loading_stages) * 0.18:
                payload = self.loading_payload
                self.loading_payload = None
                self.mode = "MENU"
                return payload
        return None

    def render(self, surface: pygame.Surface):
        """Render home UI elements on top of the transparent/3D surface."""
        f_title, f_header, f_body, f_small = self.fonts
        mx, my = pygame.mouse.get_pos()

        if self.mode == "MENU":
            cx = self.width // 2
            cy = self.height // 2

            # Central Title Panel
            panel_w = 440
            panel_h = 280
            panel_rect = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2 - 30, panel_w, panel_h)
            draw_panel(surface, panel_rect)

            title_surf = f_title.render("RLLS 16", True, ACCENT_CYAN)
            surface.blit(title_surf, title_surf.get_rect(center=(cx, cy - 115)))

            sub_surf = f_body.render("Artificial World Simulation", True, TEXT_MUTED)
            surface.blit(sub_surf, sub_surf.get_rect(center=(cx, cy - 85)))

            pygame.draw.line(surface, PANEL_BORDER, (cx - 160, cy - 65), (cx + 160, cy - 65), 1)

            btn_create, btn_load, btn_exit = self._get_menu_rects()
            draw_button(surface, btn_create, "CREATE WORLD", f_header, btn_create.collidepoint(mx, my))
            draw_button(surface, btn_load, "LOAD WORLD", f_header, btn_load.collidepoint(mx, my))
            draw_button(surface, btn_exit, "EXIT", f_header, btn_exit.collidepoint(mx, my))

            foot_surf = f_small.render("Canonical System Foundation  |  v0.3", True, TEXT_DIM)
            surface.blit(foot_surf, foot_surf.get_rect(center=(cx, self.height - 25)))

        elif self.mode == "MODAL_CREATE":
            self._render_create_modal(surface, mx, my)

        elif self.mode == "MODAL_LOAD":
            self._render_load_modal(surface, mx, my)

        elif self.mode == "LOADING":
            self._render_loading_screen(surface)

    def _render_create_modal(self, surface: pygame.Surface, mx: int, my: int):
        f_title, f_header, f_body, f_small = self.fonts
        mw, mh = 520, 560
        cx, cy = self.width // 2, self.height // 2

        # Point 37: 200 ms ease-out animation
        progress = min(1.0, (time.time() - self.modal_open_time) / 0.20)
        ease = 1.0 - (1.0 - progress) * (1.0 - progress)
        scale = 0.96 + 0.04 * ease
        alpha = int(180 * ease)

        scaled_w = int(mw * scale)
        scaled_h = int(mh * scale)
        rx = cx - scaled_w // 2
        ry = cy - scaled_h // 2

        dim_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim_surf.fill((0, 0, 0, alpha))
        surface.blit(dim_surf, (0, 0))

        modal_rect = pygame.Rect(rx, ry, scaled_w, scaled_h)
        draw_panel(surface, modal_rect, border_color=ACCENT_CYAN, fill_color=(8, 16, 30, 250))

        title_surf = f_header.render("CREATE WORLD", True, ACCENT_CYAN)
        surface.blit(title_surf, (rx + 30, ry + 20))
        sub_surf = f_small.render("Configure planetary baseline, initial habitat & cognitive policies", True, TEXT_MUTED)
        surface.blit(sub_surf, (rx + 30, ry + 44))
        pygame.draw.line(surface, PANEL_BORDER, (rx + 30, ry + 64), (rx + scaled_w - 30, ry + 64), 1)

        # 1. World Name
        name_label = f_small.render("WORLD NAME", True, TEXT_MUTED)
        surface.blit(name_label, (rx + 30, ry + 76))
        name_rect = pygame.Rect(rx + 30, ry + 95, scaled_w - 60, 28)
        name_border = ACCENT_CYAN if self.active_input == "name" else PANEL_BORDER
        pygame.draw.rect(surface, BUTTON_BG, name_rect)
        pygame.draw.rect(surface, name_border, name_rect, 1)
        txt = self.input_world_name + ("|" if self.active_input == "name" else "")
        surface.blit(f_body.render(txt, True, TEXT_PRIMARY), (name_rect.x + 8, name_rect.y + 5))

        # 2. Population
        pop_label = f_small.render("INITIAL POPULATION", True, TEXT_MUTED)
        surface.blit(pop_label, (rx + 30, ry + 132))
        pop_rect = pygame.Rect(rx + 30, ry + 152, scaled_w - 60, 28)
        pop_border = ACCENT_CYAN if self.active_input == "pop" else PANEL_BORDER
        pygame.draw.rect(surface, BUTTON_BG, pop_rect)
        pygame.draw.rect(surface, pop_border, pop_rect, 1)
        txt_pop = self.input_population + ("|" if self.active_input == "pop" else "")
        surface.blit(f_body.render(txt_pop, True, TEXT_PRIMARY), (pop_rect.x + 8, pop_rect.y + 5))

        # 3. Environment (Point 36: No "Single Select" text)
        env_label = f_small.render("INITIAL ENVIRONMENT", True, TEXT_MUTED)
        surface.blit(env_label, (rx + 30, ry + 195))

        env_start_y = ry + 215
        for i, env_name in enumerate(INITIAL_ENVIRONMENTS):
            col = i % 2
            row = i // 2
            bx = rx + 30 + col * 235
            by = env_start_y + row * 30
            b_rect = pygame.Rect(bx, by, 220, 26)

            is_sel = (i == self.selected_env_idx)
            is_hov = b_rect.collidepoint(mx, my)

            circ_x = bx + 12
            circ_y = by + 13
            pygame.draw.circle(surface, ACCENT_CYAN if is_sel else PANEL_BORDER, (circ_x, circ_y), 6, 1)
            if is_sel:
                pygame.draw.circle(surface, ACCENT_CYAN, (circ_x, circ_y), 3)

            txt_color = TEXT_PRIMARY if is_sel else (ACCENT_CYAN if is_hov else TEXT_MUTED)
            surface.blit(f_body.render(env_name, True, txt_color), (bx + 26, by + 4))

        # 4. Reinforcement Learning (Point 30, 31)
        rl_sep_y = ry + 312
        pygame.draw.line(surface, PANEL_BORDER, (rx + 30, rl_sep_y), (rx + scaled_w - 30, rl_sep_y), 1)
        rl_header = f_small.render("REINFORCEMENT LEARNING CONFIGURATION", True, ACCENT_CYAN)
        surface.blit(rl_header, (rx + 30, rl_sep_y + 8))

        rl_btn_y = ry + 340
        for i, goal in enumerate(RL_GOALS):
            gx = rx + 30 + i * 115
            g_rect = pygame.Rect(gx, rl_btn_y, 108, 26)
            is_sel = (i == self.selected_rl_goal_idx)
            draw_button(surface, g_rect, goal.upper(), f_small, g_rect.collidepoint(mx, my), is_active=is_sel)

        spec_text = "Observation: 12-dim Normalized  |  Action Space: 14 Discrete  |  Reward: Homeostatic"
        surface.blit(f_small.render(spec_text, True, TEXT_DIM), (rx + 30, ry + 376))

        # Action buttons
        btn_y = ry + scaled_h - 55
        create_rect = pygame.Rect(rx + 30, btn_y, 220, 36)
        cancel_rect = pygame.Rect(rx + 270, btn_y, 220, 36)

        draw_button(surface, create_rect, "CREATE WORLD", f_header, create_rect.collidepoint(mx, my), is_active=True)
        draw_button(surface, cancel_rect, "CANCEL", f_header, cancel_rect.collidepoint(mx, my))

    def _render_loading_screen(self, surface: pygame.Surface):
        """Point 38: Multi-stage progress checklist during world initialization."""
        f_title, f_header, f_body, f_small = self.fonts
        cx, cy = self.width // 2, self.height // 2
        pw, ph = 480, 360
        rx, ry = cx - pw // 2, cy - ph // 2

        dim_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim_surf.fill((4, 8, 16, 230))
        surface.blit(dim_surf, (0, 0))

        p_rect = pygame.Rect(rx, ry, pw, ph)
        draw_panel(surface, p_rect, border_color=ACCENT_CYAN, fill_color=(8, 16, 32, 250))

        title = f_header.render("WORLD INITIALIZATION", True, ACCENT_CYAN)
        surface.blit(title, title.get_rect(center=(cx, ry + 32)))
        pygame.draw.line(surface, PANEL_BORDER, (rx + 20, ry + 56), (rx + pw - 20, ry + 56), 1)

        elapsed = time.time() - self.loading_start_time
        active_stage = int(elapsed / 0.18)

        sy = ry + 75
        for i, stage_text in enumerate(self.loading_stages):
            if i < active_stage:
                surface.blit(f_body.render(stage_text, True, TEXT_PRIMARY), (rx + 45, sy))
                chk = f_body.render("✓", True, ACCENT_GREEN)
                surface.blit(chk, (rx + pw - 65, sy))
            elif i == active_stage:
                surface.blit(f_body.render(stage_text + " ...", True, ACCENT_CYAN), (rx + 45, sy))
                dots = f_body.render("●", True, ACCENT_AMBER)
                surface.blit(dots, (rx + pw - 65, sy))
            else:
                surface.blit(f_body.render(stage_text, True, TEXT_DIM), (rx + 45, sy))
            sy += 32

        total_stages = len(self.loading_stages)
        progress = min(1.0, elapsed / (total_stages * 0.18))
        bar_rect = pygame.Rect(rx + 40, ry + ph - 36, pw - 80, 8)
        pygame.draw.rect(surface, BUTTON_BG, bar_rect)
        pygame.draw.rect(surface, PANEL_BORDER, bar_rect, 1)
        fill_w = int((pw - 80) * progress)
        if fill_w > 0:
            pygame.draw.rect(surface, ACCENT_CYAN, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height))

    def _render_load_modal(self, surface: pygame.Surface, mx: int, my: int):
        f_title, f_header, f_body, f_small = self.fonts
        mw, mh = 480, 420
        cx, cy = self.width // 2, self.height // 2
        rx, ry = cx - mw // 2, cy - mh // 2

        dim_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim_surf.fill((0, 0, 0, 160))
        surface.blit(dim_surf, (0, 0))

        modal_rect = pygame.Rect(rx, ry, mw, mh)
        draw_panel(surface, modal_rect, border_color=ACCENT_CYAN, fill_color=(8, 16, 30, 245))

        title_surf = f_header.render("LOAD SIMULATION WORLD", True, ACCENT_CYAN)
        surface.blit(title_surf, (rx + 30, ry + 25))
        pygame.draw.line(surface, PANEL_BORDER, (rx + 30, ry + 58), (rx + mw - 30, ry + 58), 1)

        list_y = ry + 75
        if not self.saved_worlds:
            empty_surf = f_body.render("No existing simulation instances found.", True, TEXT_MUTED)
            surface.blit(empty_surf, (rx + 30, list_y + 40))
        else:
            for i, name in enumerate(self.saved_worlds):
                item_rect = pygame.Rect(rx + 30, list_y + i * 36, mw - 60, 30)
                is_sel = (i == self.selected_world_idx)
                is_hov = item_rect.collidepoint(mx, my)

                bg = BUTTON_ACTIVE if is_sel else (BUTTON_HOVER if is_hov else BUTTON_BG)
                border = ACCENT_CYAN if is_sel else PANEL_BORDER
                pygame.draw.rect(surface, bg, item_rect)
                pygame.draw.rect(surface, border, item_rect, 1)

                surface.blit(f_body.render(name, True, TEXT_PRIMARY if is_sel else TEXT_MUTED), (item_rect.x + 12, item_rect.y + 6))

        btn_y = ry + mh - 55
        load_rect = pygame.Rect(rx + 30, btn_y, 195, 36)
        cancel_rect = pygame.Rect(rx + 255, btn_y, 195, 36)

        draw_button(surface, load_rect, "LOAD WORLD", f_header, load_rect.collidepoint(mx, my), is_active=bool(self.saved_worlds))
        draw_button(surface, cancel_rect, "CANCEL", f_header, cancel_rect.collidepoint(mx, my))
