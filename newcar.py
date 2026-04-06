# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)

import math
import random
import sys
import os
import glob

import neat
import pygame

# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 60    
CAR_SIZE_Y = 60

BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit

current_generation = 0 # Generation counter
simulation_fps = 60
fullscreen_enabled = False
selected_map = "map.png"
screen = None
map_preview_cache = {}


class StopSimulation(Exception):
    pass


def apply_display_mode():
    global screen
    flags = pygame.FULLSCREEN if fullscreen_enabled else 0
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
    pygame.display.set_caption("AI Car Simulation")


def get_available_maps():
    maps = sorted(glob.glob("map*.png"))
    if not maps and os.path.exists("map.png"):
        maps = ["map.png"]
    return maps


def draw_text(surface, text, font, color, center):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=center)
    surface.blit(rendered, rect)


def get_map_preview(map_path, preview_size):
    cache_key = (map_path, preview_size)
    if cache_key not in map_preview_cache:
        preview = pygame.image.load(map_path).convert()
        preview = pygame.transform.smoothscale(preview, preview_size)
        map_preview_cache[cache_key] = preview
    return map_preview_cache[cache_key]


def show_menu():
    global fullscreen_enabled, selected_map, simulation_fps

    pygame.init()
    apply_display_mode()

    title_font = pygame.font.SysFont("Arial", 52)
    option_font = pygame.font.SysFont("Arial", 30)
    hint_font = pygame.font.SysFont("Arial", 22)

    maps = get_available_maps()
    if not maps:
        raise FileNotFoundError("No map files found. Add at least one file like map.png.")

    if selected_map not in maps:
        selected_map = maps[0]

    map_index = maps.index(selected_map)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    map_index = (map_index - 1) % len(maps)
                    selected_map = maps[map_index]
                elif event.key == pygame.K_DOWN:
                    map_index = (map_index + 1) % len(maps)
                    selected_map = maps[map_index]
                elif event.key == pygame.K_LEFT:
                    simulation_fps = max(10, simulation_fps - 10)
                elif event.key == pygame.K_RIGHT:
                    simulation_fps = min(240, simulation_fps + 10)
                elif event.key == pygame.K_f:
                    fullscreen_enabled = not fullscreen_enabled
                    apply_display_mode()
                elif event.key == pygame.K_RETURN:
                    return
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

        screen.fill((20, 22, 30))
        draw_text(screen, "AI Car Simulation", title_font, (80, 190, 230), (WIDTH // 2, 90))
        draw_text(screen, "Choose a map, preview it, then press ENTER to start", option_font, (240, 240, 240), (WIDTH // 2, 150))

        preview_rect = pygame.Rect(120, 220, 980, 550)
        info_rect = pygame.Rect(1170, 220, 620, 550)
        pygame.draw.rect(screen, (35, 38, 48), preview_rect, border_radius=12)
        pygame.draw.rect(screen, (35, 38, 48), info_rect, border_radius=12)
        pygame.draw.rect(screen, (80, 190, 230), preview_rect, 3, border_radius=12)
        pygame.draw.rect(screen, (80, 190, 230), info_rect, 3, border_radius=12)

        preview = get_map_preview(selected_map, (preview_rect.width - 30, preview_rect.height - 70))
        preview_pos = preview.get_rect(center=(preview_rect.centerx, preview_rect.centery + 20))
        screen.blit(preview, preview_pos)
        draw_text(screen, "Map Preview", option_font, (255, 255, 255), (preview_rect.centerx, preview_rect.top + 28))

        draw_text(screen, f"Selected Map: {selected_map}", option_font, (255, 220, 80), (info_rect.centerx, info_rect.top + 45))
        draw_text(screen, f"Simulation Speed: {simulation_fps} FPS", option_font, (255, 220, 80), (info_rect.centerx, info_rect.top + 95))
        draw_text(screen, f"Display: {'Fullscreen' if fullscreen_enabled else 'Windowed'}", option_font, (255, 220, 80), (info_rect.centerx, info_rect.top + 145))

        list_start_y = info_rect.top + 220
        for idx, map_name in enumerate(maps):
            color = (255, 220, 80) if idx == map_index else (210, 210, 220)
            prefix = "> " if idx == map_index else "  "
            draw_text(screen, prefix + map_name, option_font, color, (info_rect.centerx, list_start_y + idx * 42))

        draw_text(screen, "UP/DOWN: Change map", hint_font, (200, 200, 210), (info_rect.centerx, info_rect.bottom - 140))
        draw_text(screen, "LEFT/RIGHT: Change FPS", hint_font, (200, 200, 210), (info_rect.centerx, info_rect.bottom - 100))
        draw_text(screen, "F: Toggle fullscreen/windowed", hint_font, (200, 200, 210), (info_rect.centerx, info_rect.bottom - 60))
        draw_text(screen, "ENTER: Start  |  ESC: Quit", hint_font, (200, 200, 210), (info_rect.centerx, info_rect.bottom - 20))

        pygame.display.flip()

class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load('car.png').convert() # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 

        # self.position = [690, 740] # Starting Position
        self.position = [830, 920] # Starting Position
        self.angle = 0
        self.speed = 0

        self.speed_set = False # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # Calculate Center

        self.radars = [] # List For Sensors / Radars
        self.drawing_radars = [] # Radars To Be Drawn

        self.alive = True # Boolean To Check If Car is Crashed

        self.distance = 0 # Distance Driven
        self.time = 0 # Time Passed

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) # Draw Sprite
        self.draw_radar(screen) #OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        # Calculate Distance To Border And Append To Radars List
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    
    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1
        
        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def get_reward(self):
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def run_simulation(genomes, config):
    
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    info_font = pygame.font.SysFont("Arial", 18)
    paused_font = pygame.font.SysFont("Arial", 48)
    game_map = pygame.image.load(selected_map).convert() # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0
    paused = False

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    raise StopSimulation()
                elif event.key == pygame.K_f:
                    global fullscreen_enabled
                    fullscreen_enabled = not fullscreen_enabled
                    apply_display_mode()
                elif event.key == pygame.K_LEFT:
                    global simulation_fps
                    simulation_fps = max(10, simulation_fps - 10)
                elif event.key == pygame.K_RIGHT:
                    simulation_fps = min(240, simulation_fps + 10)

        if paused:
            screen.blit(game_map, (0, 0))
            for car in cars:
                if car.is_alive():
                    car.draw(screen)

            text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
            text_rect = text.get_rect()
            text_rect.center = (900, 450)
            screen.blit(text, text_rect)

            text = alive_font.render("Still Alive: " + str(sum(1 for car in cars if car.is_alive())), True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (900, 490)
            screen.blit(text, text_rect)

            text = alive_font.render("Speed: " + str(simulation_fps) + " FPS", True, (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (900, 530)
            screen.blit(text, text_rect)

            pause_surface = paused_font.render("PAUSED", True, (200, 0, 0))
            pause_rect = pause_surface.get_rect(center=(WIDTH // 2, 160))
            screen.blit(pause_surface, pause_rect)

            hint_surface = info_font.render("SPACE: Pause/Resume  LEFT/RIGHT: Speed  F: Fullscreen  ESC: Stop to Menu", True, (0, 0, 0))
            hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT - 40))
            screen.blit(hint_surface, hint_rect)

            pygame.display.flip()
            clock.tick(30)
            continue

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10 # Left
            elif choice == 1:
                car.angle -= 10 # Right
            elif choice == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 # Slow Down
            else:
                car.speed += 2 # Speed Up
        
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        text = alive_font.render("Speed: " + str(simulation_fps) + " FPS", True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 530)
        screen.blit(text, text_rect)

        hint_surface = info_font.render("SPACE: Pause/Resume  LEFT/RIGHT: Speed  F: Fullscreen  ESC: Stop to Menu", True, (0, 0, 0))
        hint_rect = hint_surface.get_rect(center=(WIDTH // 2, HEIGHT - 40))
        screen.blit(hint_surface, hint_rect)

        pygame.display.flip()
        clock.tick(simulation_fps)

if __name__ == "__main__":
    while True:
        show_menu()
        
        # Load Config
        config_path = "./config.txt"
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    config_path)

        # Create Population And Add Reporters
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        current_generation = 0
        
        # Run Simulation For A Maximum of 1000 Generations
        try:
            population.run(run_simulation, 1000)
            break
        except StopSimulation:
            continue