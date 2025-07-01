import copy
import pygame
import math
import random
from board import  boards
import numpy as np
import pickle
import os

level = copy.deepcopy(boards)

# Dimensões dinamicas para reproduzir melhor o mapa de board---
TILE_W = 28
TILE_H = 28
NUM_ROWS = len(level)
NUM_COLS = len(level[0])
WIDTH = NUM_COLS * TILE_W
HEIGHT = NUM_ROWS * TILE_H + 50 #+50 para colocar a pontuação e vida do pacman
pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])
partidas_vencidas = 0

# --- CONSTANTES GLOBAIS ---
pygame.display.set_caption("Pac-Man com Q-Learning")
timer = pygame.time.Clock()
fps = 60
font = pygame.font.Font('freesansbold.ttf', 20)
color = 'yellow'
PI = math.pi

# Definição de spawn do fantasma
INKY_SPAWN_POS_GRID = (14, 2) 
INKY_SPAWN_POS_PIXELS = (INKY_SPAWN_POS_GRID[0] * TILE_W, INKY_SPAWN_POS_GRID[1] * TILE_H)
INKY_EXIT_POS_PIXELS = (14 * TILE_W, 2 * TILE_H)

# --- HIPERPARÂMETROS DE Q-LEARNING ---
NUM_EPISODES = 40000
ALPHA = 0.1             #Taxa de aprendizagem 
GAMMA = 0.9             #Taxa de exploração
EPSILON = 1.0           #Fator desconto
EPSILON_DECAY = 0.9999  #Decaimento de exploração a cada episodio
MIN_EPSILON = 0.01      #Valor minimo que epsilon pode atingir

#Impressão dos Hiperparametros para saber valores ao rodar programa
print(f"Inicio de treinamento com seguintes hiperparametros: ALPHA: {ALPHA}   GAMMA: {GAMMA}    EPSILON: {EPSILON}")
print(f"Inicio de treinamento com seguintes hiperparametros: DECAY: {EPSILON_DECAY}   MIN_EPSILON: {MIN_EPSILON}")

q_table = {} #tabela de estados

# --- VARIÁVEIS DE ESTADO DO JOGO ---
player_x, player_y, direction = 0, 0, 0
inky_x, inky_y, inky_direction = 0, 0, 0
score, lives = 0, 3
powerup, power_counter = False, 0
eaten_ghost = []
inky_dead, inky_box = False, False
startup_counter, moving = 0, False
game_over, game_won = False, False
visited_cells = set()

# --- IMAGENS ---
PLAYER_IMG_SIZE = (int(TILE_W * 0.9), int(TILE_H * 0.9))
GHOST_IMG_SIZE = (int(TILE_W * 0.95), int(TILE_H * 0.95))
player_images = [pygame.transform.scale(pygame.image.load(f'assets/player_images/{i}.png'), PLAYER_IMG_SIZE) for i in range(1, 5)]
inky_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/blue.png'), GHOST_IMG_SIZE)
spooked_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/powerup.png'), GHOST_IMG_SIZE)
dead_img = pygame.transform.scale(pygame.image.load(f'assets/ghost_images/dead.png'), GHOST_IMG_SIZE)

class Ghost:
    def __init__(self, x_coord, y_coord, target, speed, img, direct, dead, box, id):
        self.x_pos = x_coord
        self.y_pos = y_coord
        self.center_x = self.x_pos + TILE_W // 2
        self.center_y = self.y_pos + TILE_H // 2
        self.target = target
        self.speed = speed
        self.img = img
        self.direction = direct
        self.dead = dead
        self.in_box = box
        self.id = id
        self.turns, self.in_box = self.check_collisions()
        self.rect = self.draw()

    def draw(self):
        is_eaten_in_this_powerup = eaten_ghost[self.id]
        if self.dead:
             screen.blit(dead_img, (self.x_pos, self.y_pos))
        elif powerup and not is_eaten_in_this_powerup:
             screen.blit(spooked_img, (self.x_pos, self.y_pos))
        else:
             screen.blit(self.img, (self.x_pos, self.y_pos))
        
        ghost_rect = pygame.rect.Rect((self.center_x - TILE_W//2, self.center_y - TILE_H//2), (TILE_W, TILE_H))
        return ghost_rect

    def check_collisions(self):
        self.turns = [False, False, False, False]
        pixel_check = 15 #variavel usada para verificar se o centro de um personagem irá colidir com algo quando calculado
        if 0 < self.center_x // TILE_W < (NUM_COLS - 1):
            if level[(self.center_y - pixel_check) //  TILE_H][self.center_x // TILE_W] == 9: self.turns[2] = True
            if level[self.center_y // TILE_H][(self.center_x - pixel_check) //  TILE_W] < 3 or (level[self.center_y // TILE_H][(self.center_x - pixel_check) //  TILE_W] == 9 and (self.in_box or self.dead)): self.turns[1] = True
            if level[self.center_y // TILE_H][(self.center_x + pixel_check) //  TILE_W] < 3 or (level[self.center_y // TILE_H][(self.center_x + pixel_check) //  TILE_W] == 9 and (self.in_box or self.dead)): self.turns[0] = True
            if level[(self.center_y + pixel_check) //  TILE_H][self.center_x // TILE_W] < 3 or (level[(self.center_y + pixel_check) //  TILE_H][self.center_x // TILE_W] == 9 and (self.in_box or self.dead)): self.turns[3] = True
            if level[(self.center_y - pixel_check) //  TILE_H][self.center_x // TILE_W] < 3 or (level[(self.center_y - pixel_check) //  TILE_H][self.center_x // TILE_W] == 9 and (self.in_box or self.dead)): self.turns[2] = True

            if self.direction == 2 or self.direction == 3:
                if 12 <= self.center_x % TILE_W <= 18:
                    if level[(self.center_y + pixel_check) //  TILE_H][self.center_x // TILE_W] < 3 or (level[(self.center_y + pixel_check) //  TILE_H][self.center_x // TILE_W] == 9 and (self.in_box or self.dead)): self.turns[3] = True
                    if level[(self.center_y - pixel_check) //  TILE_H][self.center_x // TILE_W] < 3 or (level[(self.center_y - pixel_check) //  TILE_H][self.center_x // TILE_W] == 9 and (self.in_box or self.dead)): self.turns[2] = True
                if 12 <= self.center_y % TILE_H <= 18:
                    if level[self.center_y // TILE_H][(self.center_x - TILE_W) // TILE_W] < 3: self.turns[1] = True
                    if level[self.center_y // TILE_H][(self.center_x + TILE_W) // TILE_W] < 3: self.turns[0] = True
            if self.direction == 0 or self.direction == 1:
                if 12 <= self.center_x % TILE_W <= 18:
                    if level[(self.center_y + TILE_H) // TILE_H][self.center_x // TILE_W] < 3: self.turns[3] = True
                    if level[(self.center_y - TILE_H) // TILE_H][self.center_x // TILE_W] < 3: self.turns[2] = True
                if 12 <= self.center_y % TILE_H <= 18:
                    if level[self.center_y // TILE_H][(self.center_x - pixel_check) //  TILE_W] < 3: self.turns[1] = True
                    if level[self.center_y // TILE_H][(self.center_x + pixel_check) //  TILE_W] < 3: self.turns[0] = True
        else:
            self.turns[0] = True
            self.turns[1] = True
            
        box_x_min, box_x_max = 5 * TILE_W, 24 * TILE_W
        box_y_min, box_y_max = 5 * TILE_H, 8 * TILE_H
        if box_x_min < self.x_pos < box_x_max and box_y_min < self.y_pos < box_y_max:
            self.in_box = True
        else:
            self.in_box = False
        return self.turns, self.in_box

    def move_inky(self): #inky é o nome do fantasma
        if self.direction == 0:
            if self.target[0] > self.x_pos and self.turns[0]: self.x_pos += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.turns[0]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                else: self.x_pos += self.speed
        elif self.direction == 1:
            if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3
            elif self.target[0] < self.x_pos and self.turns[1]: self.x_pos -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[1]:
                if self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                if self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                else: self.x_pos -= self.speed
        elif self.direction == 2:
            if self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
            elif self.target[1] < self.y_pos and self.turns[2]: self.y_pos -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] > self.y_pos and self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[3]: self.direction = 3; self.y_pos += self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[2]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                else: self.y_pos -= self.speed
        elif self.direction == 3:
            if self.target[1] > self.y_pos and self.turns[3]: self.y_pos += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.target[1] < self.y_pos and self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[2]: self.direction = 2; self.y_pos -= self.speed
                elif self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                elif self.turns[0]: self.direction = 0; self.x_pos += self.speed
            elif self.turns[3]:
                if self.target[0] > self.x_pos and self.turns[0]: self.direction = 0; self.x_pos += self.speed
                elif self.target[0] < self.x_pos and self.turns[1]: self.direction = 1; self.x_pos -= self.speed
                else: self.y_pos += self.speed
        
        if self.x_pos < -TILE_W: self.x_pos = WIDTH
        elif self.x_pos > WIDTH: self.x_pos = -TILE_W
        return self.x_pos, self.y_pos, self.direction

def draw_misc(visualizing=False):
    hud_y = HEIGHT - 25 # Posição do HUD na parte inferior
    score_text = font.render(f'Score: {score}', True, 'white')
    screen.blit(score_text, (10, hud_y))
    if powerup:
        pygame.draw.circle(screen, 'blue', (140, hud_y + 10), 10)
    for i in range(lives):
        screen.blit(pygame.transform.scale(player_images[0], (25, 25)), (WIDTH - 100 + i * 35, hud_y))
    
    if visualizing and (game_over or game_won):
        pygame.draw.rect(screen, 'white', [50, HEIGHT // 3, WIDTH - 100, HEIGHT // 4], 0, 10)
        pygame.draw.rect(screen, 'dark gray', [60, HEIGHT // 3 + 10, WIDTH - 120, HEIGHT // 4 - 20], 0, 10)
        text_str = 'Game Over! Espaço para reiniciar.' if game_over else 'Vitória! Espaço para reiniciar.'
        text_color = 'red' if game_over else 'green'
        text = font.render(text_str, True, text_color)
        text_rect = text.get_rect(center=(WIDTH / 2, HEIGHT // 3 + (HEIGHT // 4) / 2))
        screen.blit(text, text_rect)

def check_collisions(scor, power, power_count, eaten_ghosts):
    center_x = player_x + TILE_W // 2
    center_y = player_y + TILE_H // 2
    reward = 0
    
    if 0 < player_x < WIDTH - TILE_W:
        player_grid_y = center_y // TILE_H
        player_grid_x = center_x // TILE_W
        if level[player_grid_y][player_grid_x] == 1:
            level[player_grid_y][player_grid_x] = 0
            scor += 10; reward += 20
        if level[player_grid_y][player_grid_x] == 2:
            level[player_grid_y][player_grid_x] = 0
            scor += 50; power = True; power_count = 0; eaten_ghosts = [False]; reward += 50

    return scor, power, power_count, eaten_ghosts, reward

#desenha o mapa de acordo com board
def draw_board(flicker=False):
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            if level[i][j] == 1: pygame.draw.circle(screen, 'white', (j * TILE_W + (0.5 * TILE_W), i * TILE_H + (0.5 * TILE_H)), 4)
            if level[i][j] == 2 and not flicker: pygame.draw.circle(screen, 'white', (j * TILE_W + (0.5 * TILE_W), i * TILE_H + (0.5 * TILE_H)), 10)
            if level[i][j] == 3: pygame.draw.line(screen, color, (j * TILE_W + (0.5 * TILE_W), i * TILE_H), (j * TILE_W + (0.5 * TILE_W), i * TILE_H + TILE_H), 3)
            if level[i][j] == 4: pygame.draw.line(screen, color, (j * TILE_W, i * TILE_H + (0.5 * TILE_H)), (j * TILE_W + TILE_W, i * TILE_H + (0.5 * TILE_H)), 3)
            if level[i][j] == 5: pygame.draw.arc(screen, color, [(j * TILE_W - (TILE_W * 0.4)) - 2, (i * TILE_H + (0.5 * TILE_H)), TILE_W, TILE_H], 0, PI / 2, 3)
            if level[i][j] == 6: pygame.draw.arc(screen, color, [(j * TILE_W + (TILE_W * 0.5)), (i * TILE_H + (0.5 * TILE_H)), TILE_W, TILE_H], PI / 2, PI, 3)
            if level[i][j] == 7: pygame.draw.arc(screen, color, [(j * TILE_W + (TILE_W * 0.5)), (i * TILE_H - (0.4 * TILE_H)), TILE_H, TILE_H], PI, 3 * PI / 2, 3)
            if level[i][j] == 8: pygame.draw.arc(screen, color, [(j * TILE_W - (TILE_W * 0.4)) - 2, (i * TILE_H - (0.4 * TILE_H)), TILE_W, TILE_H], 3 * PI / 2, 2 * PI, 3)
            if level[i][j] == 9: pygame.draw.line(screen, 'white', (j * TILE_W, i * TILE_H + (0.5 * TILE_H)), (j * TILE_W + TILE_W, i * TILE_H + (0.5 * TILE_H)), 3)

def draw_player(counter=0):
    anim_frame = counter % (len(player_images) * 5) // 5
    if direction == 0: screen.blit(player_images[anim_frame], (player_x, player_y))
    elif direction == 1: screen.blit(pygame.transform.flip(player_images[anim_frame], True, False), (player_x, player_y))
    elif direction == 2: screen.blit(pygame.transform.rotate(player_images[anim_frame], 90), (player_x, player_y))
    elif direction == 3: screen.blit(pygame.transform.rotate(player_images[anim_frame], 270), (player_x, player_y))

def check_position(centerx, centery):
    turns = [False, False, False, False]
    pixel_check = 15 
    if 0 < centerx // TILE_W < (NUM_COLS - 1):
        if direction == 0:
            if level[centery // TILE_H][(centerx - pixel_check) //  TILE_W] < 3: turns[1] = True
        if direction == 1:
            if level[centery // TILE_H][(centerx + pixel_check) //  TILE_W] < 3: turns[0] = True
        if direction == 2:
            if level[(centery + pixel_check) //  TILE_H][centerx // TILE_W] < 3: turns[3] = True
        if direction == 3:
            if level[(centery - pixel_check) //  TILE_H][centerx // TILE_W] < 3: turns[2] = True

        if direction == 2 or direction == 3:
            if 12 <= centerx % TILE_W <= 18:
                if level[(centery + pixel_check) //  TILE_H][centerx // TILE_W] < 3: turns[3] = True
                if level[(centery - pixel_check) //  TILE_H][centerx // TILE_W] < 3: turns[2] = True
            if 12 <= centery % TILE_H <= 18:
                if level[centery // TILE_H][(centerx - TILE_W) // TILE_W] < 3: turns[1] = True
                if level[centery // TILE_H][(centerx + TILE_W) // TILE_W] < 3: turns[0] = True
        if direction == 0 or direction == 1:
            if 12 <= centerx % TILE_W <= 18:
                if level[(centery + TILE_H) // TILE_H][centerx // TILE_W] < 3: turns[3] = True
                if level[(centery - TILE_H) // TILE_H][centerx // TILE_W] < 3: turns[2] = True
            if 12 <= centery % TILE_H <= 18:
                if level[centery // TILE_H][(centerx - pixel_check) //  TILE_W] < 3: turns[1] = True
                if level[centery // TILE_H][(centerx + pixel_check) //  TILE_W] < 3: turns[0] = True
    else:
        turns[0] = True
        turns[1] = True
    return tuple(turns)

def move_player(play_x, play_y, turns_allowed):
    player_speed = 2
    if direction == 0 and turns_allowed[0]: play_x += player_speed
    elif direction == 1 and turns_allowed[1]: play_x -= player_speed
    if direction == 2 and turns_allowed[2]: play_y -= player_speed
    elif direction == 3 and turns_allowed[3]: play_y += player_speed
    return play_x, play_y

def get_state():
    player_grid_x = (player_x + TILE_W // 2) // TILE_W
    player_grid_y = (player_y + TILE_H // 2) // TILE_H

    rel_ghost_x, rel_ghost_y = 0, 0
    if not inky_dead:
        ghost_grid_x = (inky_x + TILE_W // 2) // TILE_W
        ghost_grid_y = (inky_y + TILE_H // 2) // TILE_H
        rel_ghost_x = np.sign(ghost_grid_x - player_grid_x)
        rel_ghost_y = np.sign(ghost_grid_y - player_grid_y)

    min_dist_food = float('inf')
    closest_food_pos = (0, 0)
    has_food = any(1 in row or 2 in row for row in level)
    
    if has_food:
        for i in range(NUM_ROWS):
            for j in range(NUM_COLS):
                if level[i][j] == 1 or level[i][j] == 2:
                    dist = math.hypot(player_grid_x - j, player_grid_y - i)
                    if dist < min_dist_food:
                        min_dist_food = dist
                        closest_food_pos = (j, i)
        rel_food_x = np.sign(closest_food_pos[0] - player_grid_x)
        rel_food_y = np.sign(closest_food_pos[1] - player_grid_y)
    else:
        rel_food_x, rel_food_y = 0, 0

    turns = check_position(player_x + TILE_W // 2, player_y + TILE_H // 2)
    return (rel_ghost_x, rel_ghost_y, rel_food_x, rel_food_y, turns, powerup)

def choose_action(state, allowed_turns):
    possible_actions = [i for i, can_turn in enumerate(allowed_turns) if can_turn]
    if not possible_actions: return direction
    if random.uniform(0, 1) < EPSILON:
        return random.choice(possible_actions)
    else:
        q_values = [q_table.get((state, a), 0) for a in possible_actions]
        max_q = max(q_values)
        best_actions = [a for i, a in enumerate(possible_actions) if q_values[i] == max_q]
        return random.choice(best_actions)
    
def get_target():
    # Quando o fantasma está morto, seu único alvo é o ponto de nascimento/respawn
    if inky_dead:
        return INKY_SPAWN_POS_PIXELS

    # Quando o Pac-Man tem power-up, o alvo é fugir
    if powerup:
        if player_x < WIDTH / 2: runaway_x = WIDTH
        else: runaway_x = 0
        if player_y < HEIGHT / 2: runaway_y = HEIGHT
        else: runaway_y = 0
        return (runaway_x, runaway_y)

    # Se o fantasma está vivo, sem power-up e dentro da caixa, o alvo é sair
    # Apenas para mapa grande, nesse q-learning a caixa de fantasma foi removida do mapa
    if inky_box:
        return INKY_EXIT_POS_PIXELS
    
    # Perseguir o Pac-Man se nenhuma outra prioridade foi atingida
    return (player_x, player_y)

def reset_game():
    global player_x, player_y, direction, score, lives, powerup, power_counter, eaten_ghost
    global inky_x, inky_y, inky_direction, inky_dead, game_over, game_won, level, startup_counter, visited_cells

    #Condições iniciais do pacman
    score, lives = 0, 3
    player_x, player_y = 14 * TILE_W, 10 * TILE_H
    direction = 0
    startup_counter = 0
    
    # Fantasma sempre nasce na posição de spawn definida
    inky_x, inky_y = INKY_SPAWN_POS_PIXELS
    inky_direction = 2
    
    eaten_ghost = [False]
    inky_dead = False
    powerup, power_counter = False, 0
    game_over, game_won = False, False
    visited_cells.clear()
    level = copy.deepcopy(boards)

    return get_state()

#Calcula a distância euclidiana (em grade) do Pac-Man para a comida (ponto/power-up) mais próxima.
def get_distance_to_closest_food(px, py):
    player_grid_x = (px + TILE_W // 2) // TILE_W
    player_grid_y = (py + TILE_H // 2) // TILE_H
    min_dist = float('inf')

    # Verifica se ainda existe alguma comida no tabuleiro
    has_food = any(1 in row or 2 in row for row in level)
    if not has_food:
        return 0

    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            if level[i][j] == 1 or level[i][j] == 2: # Se for um ponto ou power-up
                dist = math.hypot(player_grid_x - j, player_grid_y - i)
                if dist < min_dist:
                    min_dist = dist
    return min_dist

#Salva o estado do treinamento (Q-table, Epsilon e número do episódio) em um arquivo.
def save_checkpoint(q_table, epsilon, episode, filename="q_table_checkpoint.pkl"):
        
    checkpoint = {
        'q_table': q_table,
        'epsilon': epsilon,
        'episode': episode
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"--- Checkpoint salvo no episódio {episode + 1} em '{filename}' ---")

#Carrega a tabela, caso exista
def load_checkpoint(filename="q_table_checkpoint.pkl"):
    
    try:
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
            q_table = checkpoint['q_table']
            epsilon = checkpoint['epsilon']
            start_episode = checkpoint['episode'] + 1
            print(f"--- Checkpoint carregado. Reiniciando do episódio {start_episode}. Epsilon: {epsilon:.4f} ---")
            return q_table, epsilon, start_episode
    except FileNotFoundError:
        print("--- Nenhum checkpoint encontrado. Iniciando um novo treinamento. ---")
        return {}, 1.0, 0 # q_table vazia, epsilon inicial, começa do episódio 0
    except Exception as e:
        print(f"Erro ao carregar o checkpoint: {e}. Iniciando novo treinamento.")
        return {}, 1.0, 0

def run_game_step(action):
    global direction, player_x, player_y, score, powerup, power_counter, eaten_ghost, lives
    global game_over, game_won, moving, startup_counter
    global inky_x, inky_y, inky_direction, inky_dead, inky_box, partidas_vencidas

    # A recompensa base por passo
    reward = -1 
    
    dist_before_move = get_distance_to_closest_food(player_x, player_y)

    #Contagem de tempo de powerup do pacman
    if powerup and power_counter < 600:
        power_counter += 1
    elif powerup and power_counter >= 600:
        power_counter = 0
        powerup = False
        eaten_ghost = [False]
    
    #Leve pausa no jogo para poder iniciar, como se desse um tempo pro jogador se preparar
    moving = startup_counter >= 20
    if not moving:
        startup_counter += 1

    center_x = player_x + TILE_W // 2
    center_y = player_y + TILE_H // 2
    turns_allowed = check_position(center_x, center_y)
    
    if action in [0, 1, 2, 3] and turns_allowed[action]:
        direction = action

    if moving:
        player_x, player_y = move_player(player_x, player_y, turns_allowed)
        
        dist_after_move = get_distance_to_closest_food(player_x, player_y)
        
        # Se a distância para a comida diminuiu, pacman ganha uma recompensa
        if dist_after_move < dist_before_move:
            reward += 1.5
        # Se aumentou, uma punição para desencorajar movimentos ruins.
        # Tomar cuidado nessa lógica, pois pode causar travamento em cantos ou fazer pacman mudar de direção nos mapas
        # Em vez de jogar o jogo
        elif dist_after_move > dist_before_move:
            reward -= 2
        
        target = get_target()
        ghost_speed = 4 if inky_dead else (1 if powerup else 2)
        
        inky = Ghost(inky_x, inky_y, target, ghost_speed, inky_img, inky_direction, inky_dead, inky_box, 0)
        inky_x, inky_y, inky_direction = inky.move_inky()
        inky_box = inky.in_box
    
    score, powerup, power_counter, eaten_ghost, col_reward = check_collisions(score, powerup, power_counter, eaten_ghost)
    reward += col_reward
    player_rect = pygame.Rect(player_x, player_y, PLAYER_IMG_SIZE[0], PLAYER_IMG_SIZE[1])
    
    if 'inky' in locals() and player_rect.colliderect(inky.rect) and not inky.dead:
        if powerup:
            if not eaten_ghost[0]:
                inky_dead = True
                eaten_ghost[0] = True
                reward += 100
        else:
            lives -= 1
            reward -= 500
            if lives > 0:
                player_x, player_y = 14 * TILE_W, 10 * TILE_H
                direction = 0
                startup_counter = 0
                inky_x, inky_y = INKY_SPAWN_POS_PIXELS
                inky_direction = 2
            else:
                game_over = True
    
    if inky_dead:
        dist_to_spawn = math.hypot(inky_x - INKY_SPAWN_POS_PIXELS[0], inky_y - INKY_SPAWN_POS_PIXELS[1])
        if dist_to_spawn < (TILE_W / 2):
            inky_dead = False
            eaten_ghost[0] = False
            inky_x, inky_y = INKY_SPAWN_POS_PIXELS

    if player_x > WIDTH:
        player_x = -PLAYER_IMG_SIZE[0] + 3
    elif player_x < -PLAYER_IMG_SIZE[0]:
        player_x = WIDTH - 3
    
    if not any(1 in row or 2 in row for row in level):
        game_won = True
        partidas_vencidas += 1
        reward += 1000

    done = game_over or game_won
    new_state = get_state()
    return new_state, reward, done

def play_game_for_visualization(q_table_learned):
    print("--- Rodando episódio de visualização ---")
    state = reset_game()
    done = False
    run_visual = True
    counter = 0

    while not done and run_visual:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: run_visual = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and (game_over or game_won):
                state = reset_game(); done = False

        screen.fill('black')
        counter = (counter + 1) % 20
        draw_board(counter > 10)
        draw_player(counter)
        
        target = get_target()
        ghost_speed = 4 if inky_dead else (1 if powerup else 2)
        Ghost(inky_x, inky_y, target, ghost_speed, inky_img, inky_direction, inky_dead, inky_box, 0)
        draw_misc(visualizing=True)

        allowed_turns = check_position(player_x + TILE_W // 2, player_y + TILE_H // 2)
        possible_actions = [i for i, can_turn in enumerate(allowed_turns) if can_turn]
        
        best_action = direction
        if possible_actions:
            q_values = [q_table_learned.get((state, a), 0) for a in possible_actions]
            max_q = max(q_values)
            best_actions = [a for i, a in enumerate(possible_actions) if q_values[i] == max_q]
            best_action = random.choice(best_actions)

        state, _, done = run_game_step(best_action)
        pygame.display.flip()
        timer.tick(fps)
    
    print("--- Fim do episódio de visualização ---")


if __name__ == '__main__':

    # Nome do arquivo do checkpoint da tabela de estados
    CHECKPOINT_FILE = 'q_table_checkpoint.pkl'

    # Carrega o checkpoint ou inicia um novo
    q_table, EPSILON, start_episode = load_checkpoint(CHECKPOINT_FILE)

    print("Iniciando treinamento de Q-learning")
    total_rewards = []
    run_main = True

    while len(total_rewards) < NUM_EPISODES and run_main:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run_main = False

        episode = len(total_rewards)
        state = reset_game()
        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < 1500:
            allowed_turns = check_position(player_x + TILE_W // 2, player_y + TILE_H // 2)
            action = choose_action(state, allowed_turns)
            
            new_state, reward, done = run_game_step(action)
            episode_reward += reward

            old_value = q_table.get((state, action), 0)
            next_allowed = new_state[4] 
            next_possible = [i for i, can in enumerate(next_allowed) if can]
            next_max = max([q_table.get((new_state, a), 0) for a in next_possible]) if next_possible else 0
            
            new_q_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
            q_table[(state, action)] = new_q_value
            state = new_state
            step_count += 1

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY

        total_rewards.append(episode_reward)

        if (episode + 1) % 1 == 0:
            avg_reward = sum(total_rewards[-200:]) / 200
            print(f"Episódio: {episode + 1}/{NUM_EPISODES} | Recompensa Média: {avg_reward:.2f} | Epsilon: {EPSILON:.4f}")
            save_checkpoint(q_table, EPSILON, episode, CHECKPOINT_FILE)
            play_game_for_visualization(q_table)

    print("Treinamento concluído!")
    print(f"O pacman venceu: {partidas_vencidas} partidas")

    # Salva o estado final do treinamento
    final_episode = NUM_EPISODES -1
    save_checkpoint(q_table, EPSILON, final_episode, CHECKPOINT_FILE)
    
    # É preciso reinicializar o Pygame para a parte gráfica, caso tenha fechado a janela
    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    font = pygame.font.Font('freesansbold.ttf', 20)
    timer = pygame.time.Clock()

    while True:
        try:
            # Pede ao usuário para inserir um número
            num_visualizations_str = input("Digite o número de partidas para visualizar (ou 'sair' para fechar): ")
            
            if num_visualizations_str.lower() == 'sair':
                break

            num_visualizations = int(num_visualizations_str)
            if num_visualizations <= 0:
                print("Por favor, digite um número positivo.")
                continue

            # Roda a visualização pelo número de vezes especificado
            for i in range(num_visualizations):
                print(f"\n--- Iniciando visualização {i + 1}/{num_visualizations} ---")
                play_game_for_visualization(q_table)
            
        except ValueError:
            print("Entrada inválida. Por favor, digite um número inteiro.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            break

    pygame.quit()