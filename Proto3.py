# Import necessary libraries
import random
import string
import json
import numpy as np
from collections import defaultdict
from ursina import (
    Ursina, Entity as UrsinaEntity, Mesh, color, Tooltip, Vec3, camera, time, held_keys,
    application, Text, Button, Audio, destroy, Sequence, curve
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shape Definitions and Initialization
shapes_data = {
    'cube': {
        'vertices': [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        'edges': [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)],
        'faces': [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
    }
    # Add more shapes as needed
}

class Shape:
    def __init__(self, name, vertices, edges, faces, normals=None):
        self.name = name
        self.vertices = np.array(vertices)
        self.edges = edges
        self.faces = faces
        self.normals = np.array(normals) if normals else None
        self.lod_levels = self.generate_lod_levels()

    def generate_lod_levels(self):
        lod_levels = {
            1: self,  # High detail
            2: self.simplify_geometry(0.5),  # Mid detail
            3: self.simplify_geometry(0.25)  # Low detail
        }
        return lod_levels

    def simplify_geometry(self, reduction_factor):
        num_vertices = int(len(self.vertices) * reduction_factor)
        reduced_vertices = self.vertices[:num_vertices]
        reduced_faces = [face for face in self.faces if all(vertex < num_vertices for vertex in face)]
        return Shape(self.name + f"_lod_{int(reduction_factor*100)}", reduced_vertices, self.edges, reduced_faces)

    def get_lod_model(self, lod_level):
        return self.lod_levels.get(lod_level, self)

def load_shape_data():
    shapes = [Shape(name, **data) for name, data in shapes_data.items()]
    return shapes

# Utility Functions
projected_vertex_cache = {}

def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def frustum_cull(entity, camera_position, camera_direction, view_distance=1000):
    direction_to_entity = np.array(entity.position) - np.array(camera_position)
    distance_to_entity = np.linalg.norm(direction_to_entity)
    return distance_to_entity <= view_distance

def rotate_vertex(vertex, angle_x, angle_y, angle_z):
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    rotation_matrix_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    rotation_matrix_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    rotated_vertex = np.dot(rotation_matrix_x, vertex)
    rotated_vertex = np.dot(rotation_matrix_y, rotated_vertex)
    rotated_vertex = np.dot(rotation_matrix_z, rotated_vertex)
    return rotated_vertex

def project_vertex(vertex, screen_width, screen_height, cache_key):
    if cache_key in projected_vertex_cache:
        return projected_vertex_cache[cache_key]
    
    focal_length = 1
    projected_vertex = vertex * (screen_width / (2 * focal_length)) / vertex[2]
    projected_vertex[0] += screen_width / 2
    projected_vertex[1] = screen_height / 2 - projected_vertex[1]
    projected_vertex_cache[cache_key] = projected_vertex
    return projected_vertex

def calculate_normal(polygon):
    u = np.array(polygon[1]) - np.array(polygon[0])
    v = np.array(polygon[2]) - np.array(polygon[0])
    normal = np.cross(u, v)
    normal_length = np.linalg.norm(normal)
    return normal / normal_length if normal_length != 0 else normal

def calculate_light_intensity(normal, light_direction):
    intensity = np.dot(normal, light_direction)
    return max(0, intensity)

def create_shape_entity(shape, angle_x=0, angle_y=0, angle_z=0, light_direction=np.array([0, 0, -1]), screen_width=800, screen_height=600):
    vertices = shape.vertices
    edges = shape.edges
    faces = shape.faces

    cache_key_base = f"{angle_x}_{angle_y}_{angle_z}_{screen_width}_{screen_height}_"
    rotated_vertices = [rotate_vertex(vertex, angle_x, angle_y, angle_z) for vertex in vertices]
    projected_vertices = [project_vertex(vertex, screen_width, screen_height, cache_key_base + str(i)) for i, vertex in enumerate(rotated_vertices)]

    entities = []
    for face in faces:
        face_vertices = [projected_vertices[vertex] for vertex in face]
        normal = calculate_normal([rotated_vertices[vertex] for vertex in face])
        light_intensity = calculate_light_intensity(normal, light_direction)
        face_color = color.rgb(255 * light_intensity, 255 * light_intensity, 255 * light_intensity)
        polygon_entity = UrsinaEntity(
            model=Mesh(vertices=face_vertices, triangles=[(0, 1, 2), (2, 3, 0)], mode='triangle'),
            color=face_color
        )
        entities.append(polygon_entity)
    return entities

def determine_lod_level(distance):
    if distance < 500:
        return 1  # High detail
    elif distance < 1000:
        return 2  # Mid detail
    else:
        return 3  # Low detail

def draw_shapes(camera_position):
    shapes = load_shape_data()
    for shape in shapes:
        distance_to_camera = distance(shape.vertices.mean(axis=0), camera_position)
        lod_level = determine_lod_level(distance_to_camera)
        lod_model = shape.get_lod_model(lod_level)
        entities = create_shape_entity(lod_model)
        for entity in entities:
            entity.model = Mesh(vertices=entity.model.vertices, triangles=entity.model.triangles, mode='line')
            entity.color = color.white

# Define Interaction Zones
class InteractionZone:
    def __init__(self, name, entities, zone_type):
        self.name = name
        self.entities = entities
        self.zone_type = zone_type

def create_interaction_zones():
    zones = [
        InteractionZone("Asteroid Field", [], "resources"),
        InteractionZone("Planet Orbit", [], "markets"),
        InteractionZone("Sun Skimmer", [], "fuel_scoop"),
        # Add more zones as needed
    ]
    return zones

# Base Class for Serializable Entities
class SerializableEntity(UrsinaEntity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position = kwargs.get('position', Vec3(0, 0, 0))

    def serialize(self):
        return {
            'name': self.name,
            'position': self.position,
            'health': self.health,
            'shield': self.shield,
            'inventory': self.inventory
        }

    @staticmethod
    def deserialize(data):
        return SerializableEntity(**data)

# Space Entity Classes
class SpaceEntity(SerializableEntity):
    def __init__(self, name, position, health, shield):
        super().__init__(name=name, position=position)
        self.health = health
        self.shield = shield
        self.inventory = []

class Star(SpaceEntity):
    def __init__(self, name, position):
        super().__init__(name, position, health=1000, shield=1000)

class Planet(SpaceEntity):
    def __init__(self, name, position):
        super().__init__(name, position, health=100, shield=100)

class Asteroid(SpaceEntity):
    def __init__(self, name, position):
        super().__init__(name, position, health=50, shield=50)

class Market:
    def __init__(self, name):
        self.name = name
        self.items = defaultdict(int)

    def buy(self, item, quantity):
        self.items[item] += quantity

    def sell(self, item, quantity):
        if self.items[item] >= quantity:
            self.items[item] -= quantity
            return True
        return False

class Station(SpaceEntity):
    def __init__(self, name, position):
        super().__init__(name, position, health=500, shield=500)
        self.market = Market(name + "_Market")

# Octree for Spatial Partitioning
class OctreeNode:
    def __init__(self, center, size, depth=0):
        self.center = np.array(center)
        self.size = size
        self.depth = depth
        self.entities = []
        self.children = []

    def insert(self, entity):
        if self.children:
            index = self.get_octant(entity.position)
            self.children[index].insert(entity)
        else:
            self.entities.append(entity)
            if len(self.entities) > 10 and self.depth < 5:
                self.subdivide()
                for entity in self.entities:
                    index = self.get_octant(entity.position)
                    self.children[index].insert(entity)
                self.entities = []

    def subdivide(self):
        half_size = self.size / 2
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    offset = np.array([dx, dy, dz]) * half_size / 2
                    child_center = self.center + offset
                    self.children.append(OctreeNode(child_center, half_size, self.depth + 1))

    def get_octant(self, position):
        octant = 0
        if position[0] > self.center[0]: octant |= 1
        if position[1] > self.center[1]: octant |= 2
        if position[2] > self.center[2]: octant |= 4
        return octant

    def query_range(self, center, range_):
        result = []
        if self.children:
            for child in self.children:
                if self.aabb_intersect(center, range_, child.center, child.size):
                    result.extend(child.query_range(center, range_))
        else:
            result.extend(self.entities)
        return result

    def aabb_intersect(self, center1, size1, center2, size2):
        return all(abs(center1[i] - center2[i]) <= (size1 + size2) / 2 for i in range(3))

class Octree:
    def __init__(self, center, size, max_depth=5, max_entities=10):
        self.root = OctreeNode(center, size, depth=0)
        self.max_depth = max_depth
        self.max_entities = max_entities

    def insert(self, entity):
        self.root.insert(entity)

    def query_range(self, center, range_):
        return self.root.query_range(center, range_)

# Galaxy and System Classes
class System(SpaceEntity):
    def __init__(self, name, position):
        super().__init__(name, position, health=1000, shield=1000)
        self.planets = []
        self.stations = []
        self.asteroids = []

class Galaxy:
    def __init__(self):
        self.systems = defaultdict(list)
        self.generate_galaxy()

    def generate_galaxy(self):
        for i in range(10):
            system_name = f"System_{i}"
            position = np.random.randint(-1000, 1001, size=3)
            system = System(system_name, position)
            self.systems[system_name].append(system)

# HUD and Radar
class HUD:
    def __init__(self):
        self.health_bar = Text(text='Health: 100', position=(-0.9, 0.45), scale=2)
        self.shield_bar = Text(text='Shield: 100', position=(-0.9, 0.4), scale=2)
        self.last_health = None
        self.last_shield = None

    def update(self, player):
        if self.last_health != player.health:
            self.health_bar.text = f'Health: {player.health}'
            self.last_health = player.health
        if self.last_shield != player.shield:
            self.shield_bar.text = f'Shield: {player.shield}'
            self.last_shield = player.shield

class Radar:
    def __init__(self):
        self.radar = Text(text='Radar', position=(0.8, 0.45), scale=2)

# Player and Cockpit
class Player(SpaceEntity):
    def __init__(self, name):
        super().__init__(name, position=Vec3(0, 0, 0), health=100, shield=100)
        self.weapons = [
            Weapon(mount_point=UrsinaEntity(position=Vec3(1, 0, 0)), projectile_type=Laser),
            Weapon(mount_point=UrsinaEntity(position=Vec3(-1, 0, 0)), projectile_type=PlasmaBolt)
        ]

    def move(self, direction, speed):
        movement_vector = {
            'FORWARD': Vec3(0, 0, speed),
            'BACKWARD': Vec3(0, 0, -speed),
            'LEFT': Vec3(-speed, 0, 0),
            'RIGHT': Vec3(speed, 0, 0),
            'UP': Vec3(0, speed, 0),
            'DOWN': Vec3(0, -speed, 0),
        }
        self.position += movement_vector[direction]

    def rotate(self, axis, amount):
        if axis == 'yaw':
            self.rotation_y += amount
        elif axis == 'pitch':
            self.rotation_x += amount
        elif axis == 'roll':
            self.rotation_z += amount

    def fire_weapons(self):
        direction = Vec3(0, 0, 1)
        projectiles = []
        for weapon in self.weapons:
            projectile = weapon.fire(direction)
            if projectile:
                projectiles.append(projectile)
        return projectiles

class Cockpit:
    def __init__(self):
        self.elements = []

    def draw(self):
        self.elements.append(Text(text='Cockpit View', position=(0, 0.45), scale=2))

# Event and Mission System
class EventSystem:
    def __init__(self):
        self.events = []

    def trigger_event(self, event):
        self.events.append(event)
        logger.info(f"Event triggered: {event}")

    def process_events(self):
        while self.events:
            event = self.events.pop(0)
            print(f"Processing event: {event}")
            logger.info(f"Processing event: {event}")

class Mission:
    def __init__(self, title, description, reward):
        self.title = title
        self.description = description
        self.reward = reward
        self.is_completed = False

    def complete(self):
        self.is_completed = True
        print(f"Mission {self.title} completed! Reward: {self.reward}")
        logger.info(f"Mission {self.title} completed! Reward: {self.reward}")

class CombatMission(Mission):
    def __init__(self, title, description, reward, target):
        super().__init__(title, description, reward)
        self.target = target

class TradeMission(Mission):
    def __init__(self, title, description, reward, trade_goods):
        super().__init__(title, description, reward)
        self.trade_goods = trade_goods

class ExplorationMission(Mission):
    def __init__(self, title, description, reward, location):
        super().__init__(title, description, reward)
        self.location = location

class MissionSystem:
    def __init__(self):
        self.available_missions = []
        self.active_missions = []

    def generate_missions(self):
        self.available_missions.append(CombatMission("Eliminate Pirates", "Destroy 5 pirate ships.", 500, "Pirate"))
        self.available_missions.append(TradeMission("Deliver Goods", "Deliver 20 units of food to Station Alpha.", 300, "Food"))
        self.available_missions.append(ExplorationMission("Explore Sector", "Visit the uncharted sector.", 400, "Sector 7"))

    def accept_mission(self, mission, player):
        self.available_missions.remove(mission)
        self.active_missions.append(mission)
        print(f"Mission accepted: {mission.title}")
        logger.info(f"Mission accepted: {mission.title}")

    def complete_mission(self, mission, player):
        if mission in self.active_missions:
            mission.complete()
            self.active_missions.remove(mission)
            player.inventory.append(mission.reward)

# NPC Behavior and Management
class NPCShip(SpaceEntity):
    def __init__(self, name, position, ship_type, galaxy, player):
        super().__init__(name, position, health=100, shield=50)
        self.ship_type = ship_type
        self.galaxy = galaxy
        self.player = player
        self.inventory = []
        self.behavior = self.set_behavior()
        self.weapons = [Weapon(mount_point=UrsinaEntity(position=Vec3(1, 0, 0)), projectile_type=Laser),
                        Weapon(mount_point=UrsinaEntity(position=Vec3(-1, 0, 0)), projectile_type=PlasmaBolt)]
        self.target = None

    def set_behavior(self):
        behaviors = {
            'trader': self.trade_behavior,
            'pirate': self.patrol_behavior,
            'explorer': self.explore_behavior,
        }
        return behaviors.get(self.ship_type, self.idle_behavior)

    def patrol_behavior(self):
        self.patrol()
        if self.is_near(self.player):
            self.engage_combat(self.player)

    def trade_behavior(self):
        if self.is_near(self.player):
            self.trade_with_player()
        else:
            self.move_towards_station()

    def explore_behavior(self):
        self.explore()
        if random.random() < 0.1:
            self.collect_resources()

    def idle_behavior(self):
        pass

    def patrol(self):
        self.position += np.random.randint(-1, 2, size=3)
        if not self.target:
            self.target_nearest_enemy([self.player])

    def move_towards_station(self):
        nearest_station = self.find_nearest_station()
        self.position = list(nearest_station.position)

    def engage_combat(self, target):
        self.target = target
        if self.is_near(target):
            self.fire_weapons()

    def fire_weapons(self):
        if self.target:
            direction = (Vec3(self.target.position) - Vec3(self.position)).normalized()
            projectiles = []
            for weapon in self.weapons:
                projectile = weapon.fire(direction)
                if projectile:
                    projectiles.append(projectile)
            for projectile in projectiles:
                scene.entities.append(projectile)

    def explore(self):
        self.position += np.random.randint(-1, 2, size=3) * 5

    def collect_resources(self):
        resources = {'minerals': random.randint(0, 50), 'gas': random.randint(0, 50)}
        self.inventory.append(resources)
        print(f"{self.name} collected resources: {resources}")
        logger.info(f"{self.name} collected resources: {resources}")

    def find_nearest_station(self):
        return min(self.galaxy.systems.items(), key=lambda s: np.linalg.norm(np.array([s[0][0], s[0][1], s[0][2]]) - np.array(self.position)))

    def is_near(self, entity, range_=10):
        return np.linalg.norm(np.array(self.position) - np.array(entity.position)) < range_

    def target_nearest_enemy(self, npcs):
        if npcs:
            self.target = min(npcs, key=lambda npc: distance(self.position, npc.position))

    def update(self):
        self.behavior()

class NPCManager:
    def __init__(self):
        self.npcs = []

    def add_npc(self, npc):
        self.npcs.append(npc)

    def update_npcs(self, player):
        for npc in self.npcs:
            npc.update()
            if npc.ship_type == 'pirate':
                npc.engage_combat(player)

# Menu and UI Management
class Menu:
    def __init__(self):
        self.main_menu = None
        self.pause_menu = None
        self.inventory_menu = None
        self.transition_sequence = None

    def show_main_menu(self):
        if self.main_menu:
            self.main_menu.enable()
        else:
            self.main_menu = UrsinaEntity(parent=camera.ui, model='quad', scale=(0.8, 0.8), color=color.dark_gray)
            Button(text='New Game', parent=self.main_menu, scale=(0.3, 0.1), position=(0, 0.2), on_click=self.start_game)
            Button(text='Load Game', parent=self.main_menu, scale=(0.3, 0.1), position=(0, 0), on_click=self.load_game)
            Button(text='Quit', parent=self.main_menu, scale=(0.3, 0.1), position=(0, -0.2), on_click=application.quit)

    def start_game(self):
        self.main_menu.disable()
        scene_manager.set_scene(GalaxyScene())

    def load_game(self):
        self.main_menu.disable()
        # Implement game loading logic
        print("Game loaded")
        logger.info("Game loaded")
        scene_manager.set_scene(GalaxyScene())

    def show_pause_menu(self):
        if self.pause_menu:
            self.pause_menu.enable()
        else:
            self.pause_menu = UrsinaEntity(parent=camera.ui, model='quad', scale=(0.8, 0.8), color=color.dark_gray)
            Button(text='Resume', parent=self.pause_menu, scale=(0.3, 0.1), position=(0, 0.2), on_click=self.resume_game)
            Button(text='Save Game', parent=self.pause_menu, scale=(0.3, 0.1), position=(0, 0), on_click=self.save_game)
            Button(text='Load Game', parent=self.pause_menu, scale=(0.3, 0.1), position=(0, -0.2), on_click=self.load_game)
            Button(text='Quit', parent=self.pause_menu, scale=(0.3, 0.1), position=(0, -0.4), on_click=application.quit)

    def resume_game(self):
        self.pause_menu.disable()

    def save_game(self):
        # Implement game saving logic
        print("Game saved")
        logger.info("Game saved")

    def show_inventory(self, player):
        if self.inventory_menu:
            self.inventory_menu.enable()
        else:
            self.inventory_menu = UrsinaEntity(parent=camera.ui, model='quad', scale=(0.8, 0.8), color=color.dark_gray)
            Text(text='Inventory', parent=self.inventory_menu, position=(-0.4, 0.35), scale=2)
            y = 0.25
            for item in player.inventory:
                Text(text=item[0], parent=self.inventory_menu, position=(-0.4, y), scale=1.5)
                y -= 0.05
            Button(text='Close', parent=self.inventory_menu, scale=(0.3, 0.1), position=(0, -0.35), on_click=self.close_inventory)

    def close_inventory(self):
        self.inventory_menu.disable()

    def transition_to_scene(self, new_scene):
        if self.transition_sequence:
            self.transition_sequence.finish()
        self.transition_sequence = Sequence(1, new_scene.enable, loop=False)
        self.transition_sequence.start()

    def create_game_scene(self):
        # Implement the creation of your game scene here
        pass

class MissionInterface:
    def __init__(self):
        self.mission_menu = None
        self.available_missions_list = None
        self.active_missions_list = None

    def show_mission_interface(self, mission_system, player):
        if self.mission_menu:
            self.mission_menu.enable()
        else:
            self.mission_menu = UrsinaEntity(parent=camera.ui, model='quad', scale=(0.8, 0.8), color=color.dark_gray)
            Text(text='Available Missions', parent=self.mission_menu, position=(-0.4, 0.35), scale=2)
            self.available_missions_list = UrsinaEntity(parent=self.mission_menu, model='quad', scale=(0.7, 0.3), position=(-0.4, 0.1), color=color.light_gray)

            for i, mission in enumerate(mission_system.available_missions):
                mission_button = Button(text=f"{mission.title}: {mission.description}", parent=self.available_missions_list, scale=(0.6, 0.05), position=(-0.1, 0.25 - i * 0.06), on_click=lambda m=mission: self.accept_mission(m, mission_system, player))

            Text(text='Active Missions', parent=self.mission_menu, position=(0.4, 0.35), scale=2)
            self.active_missions_list = UrsinaEntity(parent=self.mission_menu, model='quad', scale=(0.7, 0.3), position=(0.4, 0.1), color=color.light_gray)

            for i, mission in enumerate(mission_system.active_missions):
                Text(text=f"{mission.title}: {mission.description}", parent=self.active_missions_list, position=(-0.35, 0.25 - i * 0.06), scale=1.5)

            Button(text='Close', parent=self.mission_menu, scale=(0.3, 0.1), position=(0, -0.35), on_click=self.close_mission_interface)

    def accept_mission(self, mission, mission_system, player):
        mission_system.accept_mission(mission, player)
        self.close_mission_interface()
        self.show_mission_interface(mission_system, player)

    def close_mission_interface(self):
        self.mission_menu.disable()

# Combat System
class Projectile(UrsinaEntity):
    def __init__(self, position, direction, speed=10, damage=10, lifespan=5):
        super().__init__(model='sphere', color=color.red, scale=0.1, position=position)
        self.direction = direction
        self.speed = speed
        self.damage = damage
        self.creation_time = time.time()
        self.lifespan = lifespan

    def update(self):
        self.position += self.direction * self.speed * time.dt
        if time.time() - self.creation_time > self.lifespan:
            self.disable()

class Laser(Projectile):
    def __init__(self, position, direction, speed=20, damage=15):
        super().__init__(position, direction, speed, damage)
        self.model = 'cube'
        self.scale = 0.2
        self.color = color.red

class PlasmaBolt(Projectile):
    def __init__(self, position, direction, speed=10, damage=30):
        super().__init__(position, direction, speed, damage)
        self.model = 'sphere'
        self.scale = 0.3
        self.color = color.blue

class Missile(Projectile):
    def __init__(self, position, direction, speed=5, damage=50):
        super().__init__(position, direction, speed, damage)
        self.model = 'cone'
        self.scale = 0.5
        self.color = color.yellow

class Weapon:
    def __init__(self, mount_point, fire_rate=1.0, projectile_type=Laser, projectile_speed=10, damage=10):
        self.mount_point = mount_point
        self.fire_rate = fire_rate
        self.projectile_type = projectile_type
        self.projectile_speed = projectile_speed
        self.damage = damage
        self.last_fire_time = 0

    def can_fire(self):
        return time.time() - self.last_fire_time > 1 / self.fire_rate

    def fire(self, direction):
        if self.can_fire():
            self.last_fire_time = time.time()
            projectile = self.projectile_type(position=self.mount_point.position, direction=direction, speed=self.projectile_speed, damage=self.damage)
            return projectile
        return None

def calculate_damage(weapon, target):
    damage = weapon.damage
    if target.shield > 0:
        absorbed_damage = min(target.shield, damage)
        target.shield -= absorbed_damage
        damage -= absorbed_damage
    target.health -= damage
    return damage

class CombatSystem:
    def __init__(self, player, npc_manager):
        self.player = player
        self.npc_manager = npc_manager

    def attack(self, attacker, target):
        weapon = random.choice(attacker.weapons)
        if weapon.can_fire():
            direction = (Vec3(target.position) - Vec3(attacker.position)).normalized()
            projectile = weapon.fire(direction)
            if projectile:
                scene.entities.append(projectile)
                damage = calculate_damage(weapon, target)
                print(f"{attacker.name} attacked {target.name} for {damage} damage!")
                logger.info(f"{attacker.name} attacked {target.name} for {damage} damage!")
                if target.health <= 0:
                    self.destroy(target)

    def destroy(self, target):
        print(f"{target.name} has been destroyed!")
        logger.info(f"{target.name} has been destroyed!")
        self.npc_manager.npcs.remove(target)
        target.disable()

def create_hit_effect(position):
    hit_effect = UrsinaEntity(model='sphere', color=color.yellow, scale=0.5, position=position)
    hit_effect.animate_scale(0, duration=0.5, curve=curve.in_expo)
    destroy(hit_effect, delay=0.5)
    Audio('hit_sound', loop=False).play()

def check_projectile_collisions(projectiles, targets, octree):
    potential_collisions = defaultdict(list)
    for projectile in projectiles:
        nearby_targets = octree.query_range(projectile.position, 1)
        potential_collisions[projectile].extend(nearby_targets)
    
    for projectile, potential_targets in potential_collisions.items():
        for target in potential_targets:
            if distance(projectile.position, target.position) < 1:
                if isinstance(target, SpaceEntity):
                    damage = calculate_damage(projectile, target)
                    print(f"{target.name} hit! Health: {target.health}, Shield: {target.shield}")
                    logger.info(f"{target.name} hit! Health: {target.health}, Shield: {target.shield}")
                    create_hit_effect(projectile.position)
                    projectile.disable()

# Resource System Placeholder
class ResourceSystem:
    def __init__(self):
        pass

    def collect_resources(self, entity):
        if isinstance(entity, Asteroid):
            resources = entity.resources
            print(f"Collected resources from {entity.name}: {resources}")
            logger.info(f"Collected resources from {entity.name}: {resources}")

    def trade_at_station(self, station):
        print(f"Trading at station: {station.name}")
        logger.info(f"Trading at station: {station.name}")

# Scene Management
class Scene:
    def __init__(self):
        self.entities = []

    def enter(self):
        """Called when entering the scene"""
        pass

    def exit(self):
        """Called when exiting the scene"""
        for entity in self.entities:
            destroy(entity)
        self.entities = []

    def update(self):
        """Called every frame"""
        pass

    def add_entity(self, entity):
        self.entities.append(entity)

class MainMenuScene(Scene):
    def enter(self):
        super().enter()
        self.background = UrsinaEntity(parent=camera.ui, model='quad', scale=(1.8, 1), color=color.black)
        self.title = Text("ELITE", parent=self.background, position=(0, 0.3), scale=3)
        self.new_game_button = Button(text='New Game', parent=self.background, position=(0, 0), on_click=self.start_new_game)
        self.load_game_button = Button(text='Load Game', parent=self.background, position=(0, -0.2), on_click=self.load_game)
        self.quit_button = Button(text='Quit', parent=self.background, position=(0, -0.4), on_click=application.quit)
        self.add_entity(self.background)

    def start_new_game(self):
        scene_manager.set_scene(GalaxyScene())

    def load_game(self):
        # Implement game loading logic
        print("Load game clicked")
        logger.info("Load game clicked")
        scene_manager.set_scene(GalaxyScene())

class GalaxyScene(Scene):
    def enter(self):
        super().enter()
        draw_galaxy(galaxy, octree, camera.position, camera.forward)
        self.player = player_entity
        self.add_entity(self.player)

    def update(self):
        super().update()
        camera.position = self.player.position + Vec3(0, 2, -10)
        camera.look_at(self.player.position + Vec3(0, 2, 0))

class SystemScene(Scene):
    def __init__(self, system):
        super().__init__()
        self.system = system

    def enter(self):
        super().enter()
        draw_system(self.system)
        self.player = player_entity
        self.add_entity(self.player)

    def update(self):
        super().update()
        camera.position = self.player.position + Vec3(0, 2, -10)
        camera.look_at(self.player.position + Vec3(0, 2, 0))

class CombatScene(Scene):
    def enter(self):
        super().enter()
        # Setup combat environment
        self.player = player_entity
        self.add_entity(self.player)

    def update(self):
        super().update()
        camera.position = self.player.position + Vec3(0, 2, -10)
        camera.look_at(self.player.position + Vec3(0, 2, 0))

class PlanetScene(Scene):
    def enter(self):
        super().enter()
        # Setup planet environment
        self.player = player_entity
        self.add_entity(self.player)

    def update(self):
        super().update()
        camera.position = self.player.position + Vec3(0, 2, -10)
        camera.look_at(self.player.position + Vec3(0, 2, 0))

class SceneManager:
    def __init__(self):
        self.current_scene = None

    def set_scene(self, new_scene):
        if self.current_scene:
            self.current_scene.exit()
        self.current_scene = new_scene
        self.current_scene.enter()

    def update(self):
        if self.current_scene:
            self.current_scene.update()

scene_manager = SceneManager()

# Main Game Setup
app = Ursina()

# Create a simple 3D space environment
UrsinaEntity(model='cube', color=color.black, scale=(1000, 1000, 1000))

# Create and populate a galaxy
galaxy = Galaxy()

# Initialize the octree
octree = Octree(center=[0, 0, 0], size=2000, max_depth=5, max_entities=10)

# Insert galaxy systems into the octree
for key, systems_list in galaxy.systems.items():
    for system in systems_list:
        octree.insert(system)

def draw_galaxy(galaxy, octree, camera_position, camera_direction):
    systems_to_render = octree.query_range(camera_position, 1000)
    for system in systems_to_render:
        if frustum_cull(system, camera_position, camera_direction):
            entity = UrsinaEntity(
                model='sphere', 
                color=color.green, 
                scale=2, 
                position=system.position,
                tooltip=Tooltip(f"System: {system.name}\nEconomy: {system.economy}\nPopulation: {system.population} million")
            )

camera_position = camera.position
camera_direction = camera.forward
draw_galaxy(galaxy, octree, camera_position, camera_direction)

draw_shapes(camera.position)

hud = HUD()
ui = None  # Placeholder for UI integration

player = Player("Commander")
player_entity = UrsinaEntity(model='cube', color=color.blue, scale=1, position=Vec3(0, 0, 0))
cockpit = Cockpit()
cockpit.draw()

combat_system = CombatSystem(player, NPCManager())
resource_system = ResourceSystem()  # Placeholder for Resource System

mission_system = MissionSystem()
mission_system.generate_missions()

npc_manager = NPCManager()
npc_manager.add_npc(NPCShip("Trader_1", np.random.randint(-1000, 1001, size=3), 'trader', galaxy, player))
npc_manager.add_npc(NPCShip("Pirate_1", np.random.randint(-1000, 1001, size=3), 'pirate', galaxy, player))

menu = Menu()
mission_interface = MissionInterface()
menu.show_main_menu()

def check_projectile_collisions(projectiles, targets):
    for projectile in projectiles:
        for target in targets:
            if distance(projectile.position, target.position) < 1:
                if isinstance(target, SpaceEntity) and target.shield > 0:
                    absorbed_damage = min(target.shield, projectile.damage)
                    target.shield -= absorbed_damage
                    projectile.damage -= absorbed_damage
                if projectile.damage > 0:
                    target.health -= projectile.damage
                print(f"{target.name} hit! Health: {target.health}, Shield: {target.shield}")
                logger.info(f"{target.name} hit! Health: {target.health}, Shield: {target.shield}")
                create_hit_effect(projectile.position)
                projectile.disable()

def check_interactions(player_entity, octree, interaction_range=5):
    nearby_entities = octree.query_range(player_entity.position, interaction_range)
    for entity in nearby_entities:
        if distance(player_entity.position, entity.position) < interaction_range:
            if isinstance(entity.tooltip, Tooltip):
                print(f"Interacting with {entity.tooltip.text}")
                logger.info(f"Interacting with {entity.tooltip.text}")
                if 'enemy' in entity.tooltip.text.lower():
                    combat_system.attack(player, entity)
                elif 'planet' in entity.tooltip.text.lower():
                    resource_system.collect_resources(entity)
                elif 'station' in entity.tooltip.text.lower():
                    resource_system.trade_at_station(entity)

def check_missions():
    for mission in mission_system.active_missions:
        mission_system.complete_mission(mission, player)

def input(key):
    if key == 'escape':
        menu.show_pause_menu()
    if key == 'i':
        menu.show_inventory(player)
    if key == 'm':
        mission_interface.show_mission_interface(mission_system, player)
    if key == 'g' and ui:
        ui.show_galaxy_screen(galaxy)

def update():
    scene_manager.update()
    hud.update(player)
    if ui:
        ui.show_minimap()

    speed = 5 * time.dt
    if held_keys['w']:
        player.move('FORWARD', speed)
    if held_keys['s']:
        player.move('BACKWARD', speed)
    if held_keys['a']:
        player.move('LEFT', speed)
    if held_keys['d']:
        player.move('RIGHT', speed)
    if held_keys['space']:
        player.move('UP', speed)
    if held_keys['shift']:
        player.move('DOWN', speed)

    if held_keys['q']:
        player.rotate('yaw', -0.1)
    if held_keys['e']:
        player.rotate('yaw', 0.1)
    if held_keys['r']:
        player.rotate('pitch', -0.1)
    if held_keys['f']:
        player.rotate('pitch', 0.1)
    if held_keys['z']:
        player.rotate('roll', -0.1)
    if held_keys['x']:
        player.rotate('roll', 0.1)

    player_entity.position = player.position
    player_entity.rotation = player.rotation

    camera.position = player_entity.position + Vec3(0, 2, -10)
    camera.look_at(player_entity.position + Vec3(0, 2, 0))

    if held_keys['mouse left']:
        projectiles = player.fire_weapons()
        for projectile in projectiles:
            scene.entities.append(projectile)

    check_interactions(player_entity, octree)
    check_missions()
    draw_galaxy(galaxy, octree, camera.position, camera.forward)
    npc_manager.update_npcs(player)

    projectiles = [entity for entity in scene.entities if isinstance(entity, Projectile)]
    npcs = npc_manager.npcs
    check_projectile_collisions(projectiles, npcs)
    check_projectile_collisions(projectiles, [player])

    draw_shapes(camera.position)

# Initialize the game
scene_manager.set_scene(MainMenuScene())

app.run()

