# Build Pac-Man from Scratch in Python with PyGame!!
import copy
# A variável 'boards' original foi removida daqui para ser definida abaixo com o novo layout.
import pygame
import math
### DQL: NOVAS IMPORTAÇÕES ###
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

# Determina se o CUDA (GPU) está disponível para o PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NOVO TABULEIRO ---
# 0 = empty black rectangle, 1 = dot, 2 = big dot, 3 = vertical line,
# 4 = horizontal line, 5 = top right, 6 = top left, 7 = bot left, 8 = bot right
# 9 = gate
new_board = [
    [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
    [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [3, 2, 1, 6, 4, 5, 1, 6, 4, 4, 4, 4, 4, 4, 9, 9, 4, 4, 4, 4, 4, 5, 1, 6, 4, 5, 1, 2, 1, 3],
    [3, 1, 1, 3, 0, 3, 1, 3, 6, 4, 4, 4, 5, 0, 0, 0, 0, 5, 4, 4, 4, 8, 1, 3, 0, 3, 1, 1, 1, 3],
    [3, 1, 1, 7, 4, 8, 1, 7, 4, 4, 4, 5, 3, 0, 0, 0, 0, 3, 6, 4, 4, 4, 1, 7, 4, 8, 1, 1, 1, 3],
    [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [3, 1, 1, 6, 4, 5, 1, 6, 4, 4, 4, 8, 7, 4, 4, 4, 4, 8, 7, 4, 4, 5, 1, 6, 4, 5, 1, 1, 1, 3],
    [3, 1, 1, 3, 0, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 0, 3, 1, 1, 1, 3],
    [3, 1, 1, 7, 4, 8, 1, 7, 4, 4, 4, 6, 4, 4, 9, 9, 4, 4, 5, 4, 4, 8, 1, 7, 4, 8, 1, 1, 1, 3],
    [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [3, 2, 1, 6, 4, 4, 4, 4, 4, 4, 4, 8, 1, 1, 1, 1, 1, 1, 7, 4, 4, 4, 4, 4, 4, 4, 5, 1, 2, 3],
    [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8],
]


# Garante que o caminho para os assets funcione
# Certifique-se de que a pasta 'assets' está no mesmo diretório que este script.
base_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'


pygame.init()

# --- AJUSTES DINÂMICOS DE DIMENSÕES ---
level = copy.deepcopy(new_board)
ROWS = len(level)
COLS = len(level[0])
TILE_SIZE = 28  # Tamanho de cada célula do tabuleiro em pixels
BOTTOM_MARGIN = 60 # Espaço para placar e vidas

WIDTH = COLS * TILE_SIZE
HEIGHT = ROWS * TILE_SIZE + BOTTOM_MARGIN
GAME_HEIGHT = ROWS * TILE_SIZE # Altura da área de jogo

screen = pygame.display.set_mode([WIDTH, HEIGHT])
timer = pygame.time.Clock()
fps = 60
font = pygame.font.Font('freesansbold.ttf', 20)
color = 'blue'
PI = math.pi

# --- AJUSTE NO TAMANHO DAS IMAGENS ---
IMAGE_SIZE = TILE_SIZE - 4 # Um pouco menor que o tile para caber
player_images = []
for i in range(1, 5):
    player_images.append(pygame.transform.scale(pygame.image.load(base_path + f'/assets/player_images/{i}.png'), (IMAGE_SIZE, IMAGE_SIZE)))
blinky_img = pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/red.png'), (IMAGE_SIZE, IMAGE_SIZE))
pinky_img = pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/pink.png'), (IMAGE_SIZE, IMAGE_SIZE))
inky_img = pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/blue.png'), (IMAGE_SIZE, IMAGE_SIZE))
clyde_img = pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/orange.png'), (IMAGE_SIZE, IMAGE_SIZE))
spooked_img = pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/powerup.png'), (IMAGE_SIZE, IMAGE_SIZE))
dead_img = pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/dead.png'), (IMAGE_SIZE, IMAGE_SIZE))

# --- NOVAS POSIÇÕES INICIAIS ---
# (Col, Row) -> (x, y)
# Pac-Man: (1, 1)
player_x = 1.5 * TILE_SIZE
player_y = 1.5 * TILE_SIZE
direction = 0
# Blinky (vermelho): (28, 1)
blinky_x = 28.5 * TILE_SIZE
blinky_y = 1.5 * TILE_SIZE
blinky_direction = 1
# Inky (azul): (14, 6) - dentro da caixa
inky_x = 14.5 * TILE_SIZE
inky_y = 6 * TILE_SIZE
inky_direction = 2
# Pinky (rosa): (15, 6) - dentro da caixa
pinky_x = 15.5 * TILE_SIZE
pinky_y = 6 * TILE_SIZE
pinky_direction = 2
# Clyde (laranja): (13, 6) - dentro da caixa
clyde_x = 13.5 * TILE_SIZE
clyde_y = 6 * TILE_SIZE
clyde_direction = 2

# Coordenadas da nova caixa de fantasmas para a lógica `in_box`
GHOST_BOX_X_MIN = 3 * TILE_SIZE
GHOST_BOX_X_MAX = 27 * TILE_SIZE
GHOST_BOX_Y_MIN = 3 * TILE_SIZE
GHOST_BOX_Y_MAX = 10 * TILE_SIZE
# Ponto para onde os fantasmas mortos retornam (centro da caixa)
GHOST_RETURN_TARGET = (15 * TILE_SIZE, 6 * TILE_SIZE)


counter = 0
flicker = False
turns_allowed = [False, False, False, False]
direction_command = 0
player_speed = 2
score = 0
powerup = False
power_counter = 0
eaten_ghost = [False, False, False, False]
targets = [(player_x, player_y), (player_x, player_y), (player_x, player_y), (player_x, player_y)]
blinky_dead = False
inky_dead = False
clyde_dead = False
pinky_dead = False
blinky_box = False
inky_box = False
clyde_box = False
pinky_box = False
moving = False
ghost_speeds = [2, 2, 2, 2]
startup_counter = 0
lives = 3
game_over = False
game_won = False


### Configuração de run de treino renderizada ou não:
RENDER = True
nRender = int(input("Quantas runs você deseja assistir após o treino?: "))
nTreino = int(input("Quantas runs de treino o programa deve fazer?: "))
# Contador de runs/episódios
run_counter = 0


### DQL: HIPERPARÂMETROS E CONFIGURAÇÃO DA REDE ###
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000
TARGET_UPDATE = 10
MEMORY_SIZE = 50000
LR = 1e-4
model_file = 'dqn_model_new_board.pth' # Nome de arquivo diferente para não sobrescrever o antigo

N_ACTIONS = 4
N_OBSERVATIONS = 11


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)

if os.path.exists(model_file):
    policy_net.load_state_dict(torch.load(model_file))
    print("Modelo DQL carregado do arquivo.")

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0
episode_count = 0


### DQL: FUNÇÕES PRINCIPAIS ###

def get_state(player_x, player_y, ghosts, powerup_active):
    """
    Cria uma representação vetorial do estado do jogo e a converte para um tensor PyTorch.
    Usa as dimensões do tabuleiro para normalizar as posições.
    """
    # Normaliza as posições com base nas dimensões do jogo
    px = player_x / WIDTH
    py = player_y / GAME_HEIGHT

    state = [px, py]
    
    ghost_pos = []
    for ghost in ghosts:
        gx = ghost.x_pos / WIDTH
        gy = ghost.y_pos / GAME_HEIGHT
        ghost_pos.append(gx)
        ghost_pos.append(gy)
    
    state.extend(sorted(ghost_pos)) # Ordenar para consistência
    state.append(1 if powerup_active else 0)
    
    return torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


def choose_action(state, turns_allowed, current_direction):
    """
    Decide a próxima ação usando a estratégia epsilon-greedy com a rede neural.
    Inclui uma penalidade para a ação de reverter a direção.
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    valid_actions = [i for i, allowed in enumerate(turns_allowed) if allowed]
    if not valid_actions:
        return torch.tensor([[current_direction]], device=device, dtype=torch.long)

    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            
            # Penalidade para reverter direção
            opposite_action = -1
            if current_direction == 0: opposite_action = 1
            elif current_direction == 1: opposite_action = 0
            elif current_direction == 2: opposite_action = 3
            elif current_direction == 3: opposite_action = 2

            if opposite_action != -1:
                 q_values[0][opposite_action] -= 0.5

            # Zera Q-values de ações inválidas
            for i in range(N_ACTIONS):
                if i not in valid_actions:
                    q_values[0][i] = -float('inf')
            
            return q_values.max(1)[1].view(1, 1)
    else:
        # Explora: escolhe uma ação válida aleatória
        action = random.choice(valid_actions)
        return torch.tensor([[action]], device=device, dtype=torch.long)


def optimize_model():
    """Realiza um passo de otimização na policy_net."""
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def save_model():
    """Salva o modelo DQL (os pesos da rede)."""
    torch.save(policy_net.state_dict(), model_file)
    print(f"Modelo DQL salvo. Episódio: {episode_count}, Steps: {steps_done}")


def reiniciar_jogo():
    global episode_count, powerup, power_counter, startup_counter, player_x, player_y, direction
    global direction_command, blinky_x, blinky_y, blinky_direction, inky_x, inky_y, inky_direction
    global pinky_x, pinky_y, pinky_direction, clyde_x, clyde_y, clyde_direction, eaten_ghost
    global blinky_dead, inky_dead, clyde_dead, pinky_dead, score, lives, level, game_over, game_won
    global run_counter, RENDER, nTreino, nRender

    print(f"Run {run_counter + 1}/{nTreino + nRender} - RENDER = {RENDER}")
    run_counter += 1
    if run_counter >= nTreino:
        RENDER = True
    else:
        RENDER = False

    if run_counter > nTreino + nRender:
        print("Todas as runs foram concluídas.")
        save_model()
        pygame.quit()
        exit()

    episode_count += 1
    if episode_count % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("Rede alvo atualizada.")
    if episode_count % 50 == 0:
        save_model()
    
    # Reseta as variáveis do jogo para o estado inicial do NOVO TABULEIRO
    powerup = False
    power_counter = 0
    startup_counter = 0
    player_x = 1.5 * TILE_SIZE
    player_y = 1.5 * TILE_SIZE
    direction = 0
    direction_command = 0
    blinky_x = 28.5 * TILE_SIZE
    blinky_y = 1.5 * TILE_SIZE
    blinky_direction = 1
    inky_x = 14.5 * TILE_SIZE
    inky_y = 6 * TILE_SIZE
    inky_direction = 2
    pinky_x = 15.5 * TILE_SIZE
    pinky_y = 6 * TILE_SIZE
    pinky_direction = 2
    clyde_x = 13.5 * TILE_SIZE
    clyde_y = 6 * TILE_SIZE
    clyde_direction = 2
    eaten_ghost = [False, False, False, False]
    blinky_dead = False
    inky_dead = False
    clyde_dead = False
    pinky_dead = False
    score = 0
    lives = 3
    level = copy.deepcopy(new_board)
    game_over = False
    game_won = False


class Ghost:
    def __init__(self, x_coord, y_coord, target, speed, img, direct, dead, box, id):
        self.x_pos = x_coord
        self.y_pos = y_coord
        self.center_x = self.x_pos + TILE_SIZE / 2
        self.center_y = self.y_pos + TILE_SIZE / 2
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
        # O Rect de colisão deve ser um pouco menor que o tile
        collision_rect_size = TILE_SIZE - 8
        ghost_rect = pygame.Rect((self.center_x - collision_rect_size / 2, self.center_y - collision_rect_size / 2), 
                                 (collision_rect_size, collision_rect_size))
                                 
        img_pos_x = self.x_pos + (TILE_SIZE - IMAGE_SIZE) / 2
        img_pos_y = self.y_pos + (TILE_SIZE - IMAGE_SIZE) / 2


        if RENDER:
            if (not powerup and not self.dead) or (eaten_ghost[self.id] and powerup and not self.dead):
                screen.blit(self.img, (img_pos_x, img_pos_y))
            elif powerup and not self.dead and not eaten_ghost[self.id]:
                screen.blit(spooked_img, (img_pos_x, img_pos_y))
            else:
                screen.blit(dead_img, (img_pos_x, img_pos_y))

        return ghost_rect

    def check_collisions(self):
        self.turns = [False, False, False, False] # R, L, U, D
        fudge = TILE_SIZE // 2 # Metade do tamanho do tile para checagem
        
        # Posições no grid
        col = int(self.center_x // TILE_SIZE)
        row = int(self.center_y // TILE_SIZE)

        # Checa apenas se estiver dentro dos limites do tabuleiro
        if 0 < col < COLS -1 and 0 < row < ROWS -1:
            # Verifica se está no centro de um tile para poder virar
            if self.center_x % TILE_SIZE < fudge and self.center_y % TILE_SIZE < fudge:
                # Checa se o caminho está livre (menor que 3 significa caminho, 9 é a porta)
                # Direita
                if level[row][col + 1] < 3 or (level[row][col + 1] == 9 and (self.in_box or self.dead)):
                    self.turns[0] = True
                # Esquerda
                if level[row][col - 1] < 3 or (level[row][col - 1] == 9 and (self.in_box or self.dead)):
                    self.turns[1] = True
                # Baixo
                if level[row + 1][col] < 3 or (level[row + 1][col] == 9 and (self.in_box or self.dead)):
                    self.turns[3] = True
                # Cima
                if level[row - 1][col] < 3 or (level[row - 1][col] == 9 and (self.in_box or self.dead)):
                    self.turns[2] = True

                # Mantém a direção atual se for a única opção
                if self.direction == 0: # Direita
                    self.turns[1] = False
                if self.direction == 1: # Esquerda
                    self.turns[0] = False
                if self.direction == 2: # Cima
                    self.turns[3] = False
                if self.direction == 3: # Baixo
                    self.turns[2] = False
            else: # Se não está no centro, só pode seguir reto
                if self.direction in [0, 1]: # Movendo horizontalmente
                    if level[row][col] < 3 or (level[row][col] == 9 and (self.in_box or self.dead)):
                         self.turns[self.direction] = True
                if self.direction in [2, 3]: # Movendo verticalmente
                     if level[row][col] < 3 or (level[row][col] == 9 and (self.in_box or self.dead)):
                         self.turns[self.direction] = True
        
        # Lógica para sair da caixa
        if GHOST_BOX_X_MIN < self.x_pos < GHOST_BOX_X_MAX and GHOST_BOX_Y_MIN < self.y_pos < GHOST_BOX_Y_MAX:
             self.in_box = True
             # Permite mover para cima para sair da caixa
             if level[row-1][col] == 9 or level[row-1][col] < 3 :
                 self.turns[2] = True
             if level[row+1][col] == 9 or level[row+1][col] < 3 :
                 self.turns[3] = True
             if level[row][col-1] == 9 or level[row][col-1] < 3 :
                 self.turns[1] = True
             if level[row][col+1] == 9 or level[row][col+1] < 3 :
                 self.turns[0] = True
        else:
            self.in_box = False
            
        return self.turns, self.in_box


    def move_blinky(self): # Persegue diretamente o Pac-Man
        possible_directions = []
        # r, l, u, d
        if self.turns[0]: possible_directions.append(0)
        if self.turns[1]: possible_directions.append(1)
        if self.turns[2]: possible_directions.append(2)
        if self.turns[3]: possible_directions.append(3)
        
        if len(possible_directions) > 0:
            best_dir = self.direction
            min_dist = float('inf')
            
            for direction in possible_directions:
                if direction == 0: # Direita
                    dist = math.hypot(self.x_pos + self.speed - self.target[0], self.y_pos - self.target[1])
                elif direction == 1: # Esquerda
                    dist = math.hypot(self.x_pos - self.speed - self.target[0], self.y_pos - self.target[1])
                elif direction == 2: # Cima
                    dist = math.hypot(self.x_pos - self.target[0], self.y_pos - self.speed - self.target[1])
                else: # Baixo
                    dist = math.hypot(self.x_pos - self.target[0], self.y_pos + self.speed - self.target[1])

                if dist < min_dist:
                    min_dist = dist
                    best_dir = direction
            
            self.direction = best_dir

        if self.direction == 0: self.x_pos += self.speed
        elif self.direction == 1: self.x_pos -= self.speed
        elif self.direction == 2: self.y_pos -= self.speed
        elif self.direction == 3: self.y_pos += self.speed
        
        self.center_x = self.x_pos + TILE_SIZE / 2
        self.center_y = self.y_pos + TILE_SIZE / 2
        return self.x_pos, self.y_pos, self.direction

    def move_clyde(self): # Lógica de movimento padrão/retorno para todos
        return self.move_blinky() # Simplificado para usar a mesma lógica de perseguição
    
    def move_inky(self): # Mesma lógica de Blinky por simplicidade
        return self.move_blinky()

    def move_pinky(self): # Mesma lógica de Blinky por simplicidade
        return self.move_blinky()


def draw_misc():
    if not RENDER:
        return
    score_text = font.render(f'Score: {score}', True, 'white')
    screen.blit(score_text, (10, HEIGHT - BOTTOM_MARGIN + 10))
    if powerup:
        pygame.draw.circle(screen, 'blue', (140, HEIGHT - BOTTOM_MARGIN + 25), 15)
    for i in range(lives):
        screen.blit(pygame.transform.scale(player_images[0], (30, 30)), (WIDTH - 150 + i * 40, HEIGHT - BOTTOM_MARGIN + 10))
    if game_over:
        pygame.draw.rect(screen, 'white', [50, GAME_HEIGHT/2 - 100, WIDTH - 100, 200],0, 10)
        pygame.draw.rect(screen, 'dark gray', [60, GAME_HEIGHT/2 - 90, WIDTH - 120, 180], 0, 10)
        gameover_text = font.render('GAME OVER - Reiniciando...', True, 'red')
        screen.blit(gameover_text, (WIDTH/2 - 150, GAME_HEIGHT/2 - 10))
    if game_won:
        pygame.draw.rect(screen, 'white', [50, GAME_HEIGHT/2 - 100, WIDTH - 100, 200],0, 10)
        pygame.draw.rect(screen, 'dark gray', [60, GAME_HEIGHT/2 - 90, WIDTH - 120, 180], 0, 10)
        gameover_text = font.render('VITÓRIA! - Reiniciando...', True, 'green')
        screen.blit(gameover_text, (WIDTH/2 - 150, GAME_HEIGHT/2 - 10))


def check_collisions(scor, power, power_count, eaten_ghosts):
    col = int((player_x + TILE_SIZE / 2) // TILE_SIZE)
    row = int((player_y + TILE_SIZE / 2) // TILE_SIZE)
    
    if 0 <= col < COLS and 0 <= row < ROWS:
        if level[row][col] == 1:
            level[row][col] = 0
            scor += 10
        if level[row][col] == 2:
            level[row][col] = 0
            scor += 50
            power = True
            power_count = 0
            eaten_ghosts = [False, False, False, False]
    return scor, power, power_count, eaten_ghosts


def draw_board():
    if not RENDER:
        return
    for i in range(ROWS):
        for j in range(COLS):
            x = j * TILE_SIZE
            y = i * TILE_SIZE
            if level[i][j] == 1: # Bolinha
                pygame.draw.circle(screen, 'white', (x + TILE_SIZE * 0.5, y + TILE_SIZE * 0.5), 3)
            if level[i][j] == 2 and not flicker: # Power-up
                pygame.draw.circle(screen, 'white', (x + TILE_SIZE * 0.5, y + TILE_SIZE * 0.5), 8)
            if level[i][j] == 3: # Linha vertical
                pygame.draw.line(screen, color, (x + TILE_SIZE * 0.5, y), (x + TILE_SIZE * 0.5, y + TILE_SIZE), 2)
            if level[i][j] == 4: # Linha horizontal
                pygame.draw.line(screen, color, (x, y + TILE_SIZE * 0.5), (x + TILE_SIZE, y + TILE_SIZE * 0.5), 2)
            if level[i][j] == 5: # Canto superior direito
                pygame.draw.arc(screen, color, [x - TILE_SIZE * 0.5, y, TILE_SIZE, TILE_SIZE], 0, PI / 2, 2)
            if level[i][j] == 6: # Canto superior esquerdo
                pygame.draw.arc(screen, color, [x, y, TILE_SIZE, TILE_SIZE], PI / 2, PI, 2)
            if level[i][j] == 7: # Canto inferior esquerdo
                pygame.draw.arc(screen, color, [x, y - TILE_SIZE * 0.5, TILE_SIZE, TILE_SIZE], PI, 3 * PI / 2, 2)
            if level[i][j] == 8: # Canto inferior direito
                pygame.draw.arc(screen, color, [x - TILE_SIZE * 0.5, y - TILE_SIZE * 0.5, TILE_SIZE, TILE_SIZE], 3 * PI / 2, 2 * PI, 2)
            if level[i][j] == 9: # Porta
                pygame.draw.line(screen, 'white', (x, y + TILE_SIZE * 0.5), (x + TILE_SIZE, y + TILE_SIZE * 0.5), 3)


def draw_player():
    if not RENDER:
        return
    img_pos_x = player_x + (TILE_SIZE - IMAGE_SIZE) / 2
    img_pos_y = player_y + (TILE_SIZE - IMAGE_SIZE) / 2
    
    if direction == 0: # Direita
        screen.blit(player_images[counter // 5], (img_pos_x, img_pos_y))
    elif direction == 1: # Esquerda
        screen.blit(pygame.transform.flip(player_images[counter // 5], True, False), (img_pos_x, img_pos_y))
    elif direction == 2: # Cima
        screen.blit(pygame.transform.rotate(player_images[counter // 5], 90), (img_pos_x, img_pos_y))
    elif direction == 3: # Baixo
        screen.blit(pygame.transform.rotate(player_images[counter // 5], 270), (img_pos_x, img_pos_y))


def check_position(centerx, centery):
    turns = [False, False, False, False] # R, L, U, D
    fudge = TILE_SIZE // 2
    
    col = int(centerx // TILE_SIZE)
    row = int(centery // TILE_SIZE)

    if centerx % TILE_SIZE < fudge and centery % TILE_SIZE < fudge:
        # Direita
        if col < COLS - 1 and level[row][col + 1] < 3:
            turns[0] = True
        # Esquerda
        if col > 0 and level[row][col - 1] < 3:
            turns[1] = True
        # Cima
        if row > 0 and level[row - 1][col] < 3:
            turns[2] = True
        # Baixo
        if row < ROWS - 1 and level[row + 1][col] < 3:
            turns[3] = True
            
    # Se não está alinhado, só pode seguir reto
    else:
        if direction in [0, 1]: # Horizontal
            if row > 0 and row < ROWS -1 and level[row+1][col] < 3: turns[3] = True
            if row > 0 and row < ROWS -1 and level[row-1][col] < 3: turns[2] = True
        if direction in [2, 3]: # Vertical
            if col > 0 and col < COLS - 1 and level[row][col+1] < 3: turns[0] = True
            if col > 0 and col < COLS - 1 and level[row][col-1] < 3: turns[1] = True

    return turns


def move_player(play_x, play_y):
    if direction == 0 and turns_allowed[0]: play_x += player_speed
    elif direction == 1 and turns_allowed[1]: play_x -= player_speed
    if direction == 2 and turns_allowed[2]: play_y -= player_speed
    elif direction == 3 and turns_allowed[3]: play_y += player_speed
    return play_x, play_y


def get_targets(blink_x, blink_y, ink_x, ink_y, pink_x, pink_y, clyd_x, clyd_y):
    if player_x < WIDTH / 2: runaway_x = WIDTH
    else: runaway_x = 0
    if player_y < HEIGHT / 2: runaway_y = HEIGHT
    else: runaway_y = 0

    return_target = GHOST_RETURN_TARGET

    if powerup:
        # Lógica de fuga
        if not blinky.dead: blink_target = (runaway_x, runaway_y)
        else: blink_target = return_target
        if not inky.dead: ink_target = (runaway_x, player_y)
        else: ink_target = return_target
        if not pinky.dead: pink_target = (player_x, runaway_y)
        else: pink_target = return_target
        if not clyde.dead: clyd_target = (450, 450) # Clyde ainda pode ter comportamento diferente
        else: clyd_target = return_target
    else:
        # Lógica de perseguição
        if not blinky.dead: blink_target = (player_x, player_y)
        else: blink_target = return_target
        if not inky.dead: ink_target = (player_x, player_y)
        else: ink_target = return_target
        if not pinky.dead: pink_target = (player_x, player_y)
        else: pink_target = return_target
        if not clyde.dead: clyd_target = (player_x, player_y)
        else: clyd_target = return_target
        
    return [blink_target, ink_target, pink_target, clyd_target]


def find_closest_pellet(player_x, player_y, current_level):
    min_dist = float('inf')
    for i in range(ROWS):
        for j in range(COLS):
            if current_level[i][j] in [1, 2]:
                pellet_x = j * TILE_SIZE + (0.5 * TILE_SIZE)
                pellet_y = i * TILE_SIZE + (0.5 * TILE_SIZE)
                dist = math.hypot(player_x - pellet_x, player_y - pellet_y)
                if dist < min_dist:
                    min_dist = dist
    return min_dist if min_dist != float('inf') else 0


# Loop principal
run = True
while run:
    if RENDER:
        timer.tick(fps)
    if counter < 19:
        counter += 1
        if counter > 3: flicker = False
    else:
        counter = 0
        flicker = True
    if powerup and power_counter < 600:
        power_counter += 1
    elif powerup and power_counter >= 600:
        power_counter = 0
        powerup = False
        eaten_ghost = [False, False, False, False]
    if startup_counter < 180 and not game_over and not game_won:
        moving = False
        startup_counter += 1
    else:
        moving = True

    screen.fill('black')
    draw_board()
    
    # Centro do jogador para colisões
    center_x = player_x + TILE_SIZE / 2
    center_y = player_y + TILE_SIZE / 2
    
    score_before_action = score
    
    if powerup: ghost_speeds = [1, 1, 1, 1]
    else: ghost_speeds = [2, 2, 2, 2]
    
    if blinky_dead: ghost_speeds[0] = 4
    if inky_dead: ghost_speeds[1] = 4
    if pinky_dead: ghost_speeds[2] = 4
    if clyde_dead: ghost_speeds[3] = 4

    game_won = True
    for i in range(len(level)):
        if 1 in level[i] or 2 in level[i]:
            game_won = False
            break

    collision_rect_size = TILE_SIZE - 8
    player_rect = pygame.Rect(center_x - collision_rect_size/2, center_y - collision_rect_size/2, collision_rect_size, collision_rect_size)
    
    draw_player()
    
    blinky = Ghost(blinky_x, blinky_y, targets[0], ghost_speeds[0], blinky_img, blinky_direction, blinky_dead, blinky_box, 0)
    inky = Ghost(inky_x, inky_y, targets[1], ghost_speeds[1], inky_img, inky_direction, inky_dead, inky_box, 1)
    pinky = Ghost(pinky_x, pinky_y, targets[2], ghost_speeds[2], pinky_img, pinky_direction, pinky_dead, pinky_box, 2)
    clyde = Ghost(clyde_x, clyde_y, targets[3], ghost_speeds[3], clyde_img, clyde_direction, clyde_dead, clyde_box, 3)
    
    all_ghosts = [blinky, inky, pinky, clyde]
    draw_misc()
    targets = get_targets(blinky.x_pos, blinky.y_pos, inky.x_pos, inky.y_pos, pinky.x_pos, pinky.y_pos, clyde.x_pos, clyde.y_pos)

    if not game_over and not game_won:
        state = get_state(player_x, player_y, all_ghosts, powerup)
        
        turns_allowed = check_position(center_x, center_y)
        
        action_tensor = choose_action(state, turns_allowed, direction)
        action = action_tensor.item()
        direction_command = action

        if direction_command == 0 and turns_allowed[0]: direction = 0
        if direction_command == 1 and turns_allowed[1]: direction = 1
        if direction_command == 2 and turns_allowed[2]: direction = 2
        if direction_command == 3 and turns_allowed[3]: direction = 3

        old_player_x, old_player_y = player_x, player_y

        if moving:
            player_x, player_y = move_player(player_x, player_y)
            blinky_x, blinky_y, blinky_direction = blinky.move_blinky()
            pinky_x, pinky_y, pinky_direction = pinky.move_pinky()
            inky_x, inky_y, inky_direction = inky.move_inky()
            clyde_x, clyde_y, clyde_direction = clyde.move_clyde()
        
        score, powerup, power_counter, eaten_ghost = check_collisions(score, powerup, power_counter, eaten_ghost)
        
        player_lost_life = False
        ghost_collided = -1
        if not powerup:
            for i, ghost in enumerate(all_ghosts):
                if player_rect.colliderect(ghost.rect) and not ghost.dead:
                    player_lost_life = True
                    lives -= 1
                    if lives <= 0: game_over = True
                    break 
        else:
             for i, ghost in enumerate(all_ghosts):
                if player_rect.colliderect(ghost.rect) and not ghost.dead and not eaten_ghost[i]:
                    ghost_collided = i
                    break
        
        if ghost_collided != -1:
            if ghost_collided == 0: blinky_dead = True
            elif ghost_collided == 1: inky_dead = True
            elif ghost_collided == 2: pinky_dead = True
            elif ghost_collided == 3: clyde_dead = True
            
            eaten_ghost[ghost_collided] = True
            score += (2 ** eaten_ghost.count(True)) * 100

        # Lógica de Recompensa
        reward = (score - score_before_action) * 10

        dist_before = find_closest_pellet(old_player_x, old_player_y, level)
        dist_after = find_closest_pellet(player_x, player_y, level)

        if dist_after < dist_before: reward += 1
        else: reward -= 1.5

        if player_lost_life: reward -= 500
        if game_won: reward += 1000
        reward -= 1 # Penalidade de tempo

        reward_tensor = torch.tensor([reward], device=device)

        done = game_over or game_won
        if done: next_state = None
        else: next_state = get_state(player_x, player_y, all_ghosts, powerup)

        memory.push(state, action_tensor, next_state, reward_tensor)
        optimize_model()

        if player_lost_life and not game_over:
            startup_counter = 0
            powerup = False
            power_counter = 0
            player_x = 1.5 * TILE_SIZE
            player_y = 1.5 * TILE_SIZE
            direction = 0
            direction_command = 0
            blinky_x = 28.5 * TILE_SIZE
            blinky_y = 1.5 * TILE_SIZE
            blinky_direction = 1
            inky_x = 14.5 * TILE_SIZE
            inky_y = 6 * TILE_SIZE
            inky_direction = 2
            pinky_x = 15.5 * TILE_SIZE
            pinky_y = 6 * TILE_SIZE
            pinky_direction = 2
            clyde_x = 13.5 * TILE_SIZE
            clyde_y = 6 * TILE_SIZE
            clyde_direction = 2
            eaten_ghost = [False, False, False, False]
            blinky_dead = False
            inky_dead = False
            clyde_dead = False
            pinky_dead = False

    if game_over or game_won:
        if RENDER:
            draw_misc()
            pygame.display.flip()
            pygame.time.delay(1000)
        reiniciar_jogo()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Resetar fantasmas quando chegam na caixa
    if blinky.in_box and blinky_dead: blinky_dead = False
    if inky.in_box and inky_dead: inky_dead = False
    if pinky.in_box and pinky_dead: pinky_dead = False
    if clyde.in_box and clyde_dead: clyde_dead = False

    pygame.display.flip()

save_model()
pygame.quit()