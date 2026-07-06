# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

import numpy as np
from isaacgym import terrain_utils
from numpy.random import choice

from aliengo_gym.envs.base.legged_robot_config import BaseCfg as Cfg


class Terrain:
    def __init__(self, cfg: Cfg.terrain, num_robots, eval_cfg=None, num_eval_robots=0) -> None:

        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()
        self.tot_rows = len(self.train_rows) + len(self.eval_rows)
        self.tot_cols = max(len(self.train_cols), len(self.eval_cols))
        self.cfg.env_length = cfg.terrain_length
        self.cfg.env_width = cfg.terrain_width

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.initialize_terrains()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def load_cfgs(self):
        self._load_cfg(self.cfg)
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)
        self.cfg.x_offset = 0
        self.cfg.rows_offset = 0
        if self.eval_cfg is None:
            return self.cfg.row_indices, self.cfg.col_indices, [], []
        else:
            self._load_cfg(self.eval_cfg)
            self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows, self.cfg.tot_rows + self.eval_cfg.tot_rows)
            self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)
            self.eval_cfg.x_offset = self.cfg.tot_rows
            self.eval_cfg.rows_offset = self.cfg.num_rows
            return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices

    def _load_cfg(self, cfg):
        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        cfg.length_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        cfg.width_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)

        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border

    def initialize_terrains(self):
        self._initialize_terrain(self.cfg)
        if self.eval_cfg is not None:
            self._initialize_terrain(self.eval_cfg)

    def _initialize_terrain(self, cfg):
        if cfg.curriculum:
            self.curriculum(cfg)
        elif cfg.selected:
            self.selected_terrain(cfg)
        else:
            self.randomized_terrain(cfg)

    def randomized_terrain(self, cfg):
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def curriculum(self, cfg):
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                # difficulty = i / cfg.num_rows * cfg.difficulty_scale
                # With ten rows, the highest difficulty is only 0.9, not 1.0
                # row 0 -> difficulty 0.0
                # row 9 -> difficulty 1.0
                difficulty = i / max(cfg.num_rows - 1, 1) * cfg.difficulty_scale
                choice = j / cfg.num_cols + 0.001

                terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
                self.add_terrain_to_map(cfg, terrain, i, j)

    def selected_terrain(self, cfg):
        terrain_type = cfg.terrain_kwargs.pop('type')
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=cfg.width_per_env_pixels,
                length=cfg.length_per_env_pixels,
                vertical_scale=cfg.vertical_scale,
                horizontal_scale=cfg.horizontal_scale,
            )

            eval(terrain_type)(terrain, **cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def gap_terrain(terrain, gap_width, gap_depth=0.5, platform_size=1.2):
        """
        Creates a square gap surrounding a central safe platform.

        Args:
            terrain:
                Isaac Gym SubTerrain.
            gap_width:
                Width of the gap surrounding the platform [m].
            gap_depth:
                Depth of the gap [m].
            platform_size:
                Width of the central platform [m].
        """

        horizontal_scale = terrain.horizontal_scale
        vertical_scale = terrain.vertical_scale

        gap_px = max(1, int(round(gap_width / horizontal_scale)))

        platform_px = max(2, int(round(platform_size / horizontal_scale)))

        depth_raw = int(round(-abs(gap_depth) / vertical_scale))

        center_x = terrain.length // 2
        center_y = terrain.width // 2

        platform_half = platform_px // 2
        outer_half = platform_half + gap_px

        outer_x1 = max(0, center_x - outer_half)
        outer_x2 = min(terrain.length, center_x + outer_half)

        outer_y1 = max(0, center_y - outer_half)
        outer_y2 = min(terrain.width, center_y + outer_half)

        platform_x1 = max(0, center_x - platform_half)
        platform_x2 = min(terrain.length, center_x + platform_half)

        platform_y1 = max(0, center_y - platform_half)
        platform_y2 = min(terrain.width, center_y + platform_half)

        # Dig the outer square.
        terrain.height_field_raw[outer_x1:outer_x2, outer_y1:outer_y2] = depth_raw

        # Restore the central spawn platform.
        terrain.height_field_raw[platform_x1:platform_x2, platform_y1:platform_y2] = 0

        return terrain

    def pillar_terrain(terrain, pillar_size, pillar_spacing, pillar_height, center_platform_size=1.2):
        """
        Creates a grid of raised square pillars with a raised central
        platform on which the robot is initialized.

        Args:
            terrain:
                Isaac Gym SubTerrain.
            pillar_size:
                Width of each pillar [m].
            pillar_spacing:
                Horizontal gap between pillars [m].
            pillar_height:
                Pillar height [m].
            center_platform_size:
                Width of the central spawn platform [m].
        """

        horizontal_scale = terrain.horizontal_scale
        vertical_scale = terrain.vertical_scale

        pillar_px = max(1, int(round(pillar_size / horizontal_scale)))

        spacing_px = max(1, int(round(pillar_spacing / horizontal_scale)))

        center_platform_px = max(pillar_px, int(round(center_platform_size / horizontal_scale)))

        pillar_height_raw = max(1, int(round(pillar_height / vertical_scale)))

        stride = pillar_px + spacing_px

        center_x = terrain.length // 2
        center_y = terrain.width // 2
        center_half = center_platform_px // 2

        center_x1 = max(0, center_x - center_half)
        center_x2 = min(terrain.length, center_x + center_half)

        center_y1 = max(0, center_y - center_half)
        center_y2 = min(terrain.width, center_y + center_half)

        # Create a regular field of pillars.
        for x1 in range(0, terrain.length - pillar_px + 1, stride):
            x2 = x1 + pillar_px

            for y1 in range(0, terrain.width - pillar_px + 1, stride):
                y2 = y1 + pillar_px

                # Do not overwrite the central spawn region.
                overlaps_center = not (
                    x2 <= center_x1
                    or x1 >= center_x2
                    or y2 <= center_y1
                    or y1 >= center_y2
                )

                if overlaps_center:
                    continue

                terrain.height_field_raw[x1:x2, y1:y2] = pillar_height_raw

        # Raised central platform guarantees a valid spawn surface.
        terrain.height_field_raw[center_x1:center_x2, center_y1:center_y2] = pillar_height_raw

        return terrain

    def make_terrain(self, cfg, choice, difficulty, proportions):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=cfg.width_per_env_pixels,
            length=cfg.length_per_env_pixels,
            vertical_scale=cfg.vertical_scale,
            horizontal_scale=cfg.horizontal_scale,
        )

        # Normalized curriculum difficulty.
        difficulty = float(np.clip(difficulty, 0.0, 1.0))

        def lerp(easy, hard):
            """Linear interpolation from easy at d=0 to hard at d=1."""
            return easy + difficulty * (hard - easy)

        # ------------------------------------------------------------
        # Difficulty-dependent terrain parameters
        # ------------------------------------------------------------

        # Smooth and rough slopes: 0% -> 40% gradient.
        slope = lerp(0.0, 0.40)

        # Rough component added to the rough slope.
        rough_slope_height = lerp(cfg.vertical_scale, 0.05)

        # Stairs: 5 cm -> 23 cm.
        step_height = lerp(0.05, 0.23)

        # Discrete obstacles:
        # progressively increase both height and obstacle density.
        obstacle_height = lerp(0.03, cfg.max_platform_height)
        num_rectangles = int(round(lerp(8, 20)))

        # Stepping stones:
        # progressively smaller stones, larger gaps and uneven heights.
        stepping_stones_size = lerp(0.45, 0.20)
        stone_distance = lerp(0.05, 0.30)
        stepping_stones_height = lerp(0.0, 0.04)

        stepping_stones_size = max(stepping_stones_size, 2.0 * cfg.horizontal_scale)
        stone_distance = max(stone_distance, cfg.horizontal_scale)

        # Fully random terrain.
        random_noise_height = lerp(cfg.vertical_scale, cfg.terrain_noise_magnitude)

        # Rough half of the half-flat/half-rough terrain.
        half_rough_height = lerp(cfg.vertical_scale, 0.05)

        height_step = max(cfg.vertical_scale, getattr(cfg, "terrain_smoothness", cfg.vertical_scale))

        # ------------------------------------------------------------
        # Gap curriculum
        # ------------------------------------------------------------

        # Wider and deeper gaps at higher terrain levels.
        gap_width = lerp(0.05, 0.35)
        gap_depth = lerp(0.15, 0.60)

        # Reduce the central support area as difficulty increases.
        gap_platform_size = lerp(1.40, 0.90)


        # ------------------------------------------------------------
        # Pillar curriculum
        # ------------------------------------------------------------

        # Pillars become smaller, farther apart and taller.
        pillar_size = lerp(0.45, 0.20)
        pillar_spacing = lerp(0.05, 0.25)
        pillar_height = lerp(0.03, 0.25)

        # Reduce the central support platform gradually.
        pillar_center_platform = lerp(1.40, 0.90)

        # ------------------------------------------------------------
        # Terrain selection
        # ------------------------------------------------------------

        if choice < proportions[0]:
            # Smooth ascending/descending slope.
            if choice < proportions[0] / 2:
                slope *= -1.0

            terrain_utils.pyramid_sloped_terrain(
                terrain,
                slope=slope,
                platform_size=0.5,
            )

        elif choice < proportions[1]:
            # Rough slope: both slope and roughness increase.
            terrain_utils.pyramid_sloped_terrain(
                terrain,
                slope=slope,
                platform_size=0.5,
            )

            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_slope_height,
                max_height=rough_slope_height,
                step=height_step,
                downsampled_scale=0.2,
            )

        elif choice < proportions[3]:
            # Ascending/descending stairs.
            if choice < proportions[2]:
                step_height *= -1.0

            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.31,
                step_height=step_height,
                platform_size=0.8,
            )

        elif choice < proportions[4]:
            # Discrete obstacles.
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                max_height=obstacle_height,
                min_size=1.0,
                max_size=2.0,
                num_rects=num_rectangles,
                platform_size=0.8,
            )

        elif choice < proportions[5]:
            # Stepping stones.
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=stepping_stones_height,
                platform_size=0.8,
            )

        elif choice < proportions[6]:
            gap_terrain(
                terrain,
                gap_width=gap_width,
                gap_depth=gap_depth,
                platform_size=gap_platform_size,
            )

        elif choice < proportions[7]:
            pillar_terrain(
                terrain,
                pillar_size=pillar_size,
                pillar_spacing=pillar_spacing,
                pillar_height=pillar_height,
                center_platform_size=pillar_center_platform,
            )

        elif choice < proportions[8]:
            # Random rough terrain.
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-random_noise_height,
                max_height=random_noise_height,
                step=height_step,
                downsampled_scale=0.2,
            )

        elif choice < proportions[9]:
            # Half-flat, half-rough terrain.
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-half_rough_height,
                max_height=half_rough_height,
                step=height_step,
                downsampled_scale=0.2,
            )

            terrain.height_field_raw[0:terrain.length // 2, :] = 0

        return terrain

    def add_terrain_to_map(self, cfg, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        x1 = int((cfg.terrain_length / 2. - 1) / terrain.horizontal_scale) + cfg.x_offset
        x2 = int((cfg.terrain_length / 2. + 1) / terrain.horizontal_scale) + cfg.x_offset
        y1 = int((cfg.terrain_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((cfg.terrain_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(self.height_field_raw[start_x: end_x, start_y:end_y]) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
