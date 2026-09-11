import pygame

# -------------------------------------------------------------
# RLLS 16 Visual Language & Color Palette
# -------------------------------------------------------------
BG_DARK = (4, 8, 18)
PANEL_BG = (10, 20, 36, 230)
PANEL_BORDER = (38, 92, 150)
PANEL_BORDER_FOCUS = (46, 216, 232)

ACCENT_CYAN = (46, 216, 232)
ACCENT_BLUE = (56, 159, 255)
ACCENT_AMBER = (245, 175, 45)
ACCENT_GREEN = (40, 215, 120)

TEXT_PRIMARY = (235, 245, 255)
TEXT_MUTED = (135, 170, 200)
TEXT_DIM = (75, 105, 135)

BUTTON_BG = (14, 30, 54)
BUTTON_HOVER = (22, 48, 86)
BUTTON_ACTIVE = (30, 75, 135)
BUTTON_BORDER = (45, 110, 180)


def get_fonts():
    # Crisp technical monospace typography
    font_title = pygame.font.SysFont("consolas", 22, bold=True)
    font_header = pygame.font.SysFont("consolas", 16, bold=True)
    font_body = pygame.font.SysFont("consolas", 13)
    font_small = pygame.font.SysFont("consolas", 11)
    return font_title, font_header, font_body, font_small


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, border_color=PANEL_BORDER, fill_color=PANEL_BG):
    """Draw futuristic sci-fi workstation panel with subtle corner bevels."""
    panel_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel_surf.fill(fill_color)
    surface.blit(panel_surf, (rect.x, rect.y))
    pygame.draw.rect(surface, border_color, rect, width=1)


def draw_button(surface: pygame.Surface, rect: pygame.Rect, text: str, font, is_hover: bool = False, is_active: bool = False, active_color=ACCENT_CYAN):
    bg = BUTTON_ACTIVE if is_active else (BUTTON_HOVER if is_hover else BUTTON_BG)
    border = active_color if is_active else (ACCENT_CYAN if is_hover else BUTTON_BORDER)
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, border, rect, width=1)

    txt_surf = font.render(text, True, TEXT_PRIMARY if (is_hover or is_active) else TEXT_MUTED)
    txt_rect = txt_surf.get_rect(center=rect.center)
    surface.blit(txt_surf, txt_rect)


def draw_gps_icon(surface: pygame.Surface, cx: int, cy: int, size: int = 14, color=ACCENT_CYAN):
    """Draw a proper technical location/GPS crosshair icon (no emojis)."""
    r = size // 2
    # Outer circle
    pygame.draw.circle(surface, color, (cx, cy), r, width=1)
    # Center dot
    pygame.draw.circle(surface, color, (cx, cy), 2)
    # 4 crosshair tick marks extending outwards
    tick = 3
    pygame.draw.line(surface, color, (cx - r - tick, cy), (cx - r + 1, cy), 1)
    pygame.draw.line(surface, color, (cx + r - 1, cy), (cx + r + tick, cy), 1)
    pygame.draw.line(surface, color, (cx, cy - r - tick), (cx, cy - r + 1), 1)
    pygame.draw.line(surface, color, (cx, cy + r - 1), (cx, cy + r + tick), 1)


def draw_chevron(surface: pygame.Surface, x: int, y: int, is_expanded: bool, color=TEXT_MUTED):
    """Draw clean collapsible accordion indicator."""
    if is_expanded:
        pts = [(x, y), (x + 8, y), (x + 4, y + 5)]
    else:
        pts = [(x, y), (x + 5, y + 4), (x, y + 8)]
    pygame.draw.polygon(surface, color, pts)
