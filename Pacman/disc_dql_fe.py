# Pac-Man com DQL - Versão Discreta com Sprites
# Refatorado por Gemini
import pygame
import math
import random
import os
import copy
from collections import deque, namedtuple

# --- DRL Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Configurações Iniciais e Constantes ---
WIDTH = 900
HEIGHT = 950
SCREEN = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Pac-Man DQL Discreto")

# --- Constantes do Grid ---
# O tabuleiro tem 30 colunas e 31 linhas de jogo (sem contar o placar)
TILE_WIDTH = WIDTH // 30
TILE_HEIGHT = (HEIGHT - 50) // 32

# --- Carregamento de Assets (Imagens e Fontes) ---
def load_assets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    assets = {
        'font': pygame.font.Font('freesansbold.ttf', 20),
        'player': [pygame.transform.scale(pygame.image.load(base_path + f'/assets/player_images/{i}.png'), (30, 30)) for i in range(1, 5)],
        'blinky': pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/red.png'), (30, 30)),
        'pinky': pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/pink.png'), (30, 30)),
        'inky': pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/blue.png'), (30, 30)),
        'clyde': pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/orange.png'), (30, 30)),
        'spooked': pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/powerup.png'), (30, 30)),
        'dead': pygame.transform.scale(pygame.image.load(base_path + f'/assets/ghost_images/dead.png'), (30, 30)),
    }
    return assets

# --- Classes de Sprites ---
class Player(pygame.sprite.Sprite):
    def __init__(self, assets, initial_pos):
        super().__init__()
        self.assets = assets
        self.images = self.assets['player']
        self.image_idx = 0
        self.image = self.images[self.image_idx]
        self.row, self.col = initial_pos
        self.rect = self.image.get_rect(center=self.get_pixel_pos())
        self.direction = 0  # 0:R, 1:L, 2:U, 3:D
        self.animation_counter = 0
        self.lives = 3

    def update(self, action, level_board):
        self.direction = action
        next_row, next_col = self.get_next_grid_pos(action)

        # Verifica se o próximo movimento é válido (não é uma parede)
        if level_board[next_row][next_col] < 3:
            self.row, self.col = next_row, next_col
        
        # Atualiza a posição em pixels e a animação
        self.rect.center = self.get_pixel_pos()
        self.animate()

    def animate(self):
        self.animation_counter = (self.animation_counter + 1) % 20
        self.image_idx = self.animation_counter // 5
        
        base_image = self.images[self.image_idx]
        if self.direction == 1: # Left
            self.image = pygame.transform.flip(base_image, True, False)
        elif self.direction == 2: # Up
            self.image = pygame.transform.rotate(base_image, 90)
        elif self.direction == 3: # Down
            self.image = pygame.transform.rotate(base_image, 270)
        else: # Right
            self.image = base_image

    def get_pixel_pos(self):
        return (self.col * TILE_WIDTH + TILE_WIDTH // 2, self.row * TILE_HEIGHT + TILE_HEIGHT // 2)

    def get_next_grid_pos(self, action):
        if action == 0: return self.row, self.col + 1
        if action == 1: return self.row, self.col - 1
        if action == 2: return self.row - 1, self.col
        if action == 3: return self.row + 1, self.col
        return self.row, self.col
    
    def handle_teleport(self):
        if self.col < -1: self.col = 29
        elif self.col > 29: self.col = -1

### SUBSTITUA A CLASSE GHOST INTEIRA POR ESTA ###

class Ghost(pygame.sprite.Sprite):
    def __init__(self, assets, initial_pos, ghost_type, ghost_id):
        super().__init__()
        self.assets = assets
        self.ghost_type = ghost_type
        self.id = ghost_id
        self.row, self.col = initial_pos
        self.home_pos = initial_pos # Guarda a posição inicial para resetar
        self.image = self.assets[self.ghost_type]
        self.rect = self.image.get_rect(center=self.get_pixel_pos())
        self.direction = 0
        self.is_dead = False
        self.target = None
        self.is_in_box = True # Começa na caixa

    def update(self, player, blinky, powerup_active, level_board):
        # 1. Definir o alvo com base no estado do jogo
        if self.is_dead:
            self.image = self.assets['dead']
            # O alvo é a entrada da caixa de fantasmas
            self.target = (13, 14) 
        elif powerup_active:
            self.image = self.assets['spooked']
            # Lógica de fuga: ir para um canto longe do jogador
            if player.col > 15 and player.row > 15: self.target = (1, 1) # Canto superior esquerdo
            elif player.col > 15 and player.row <= 15: self.target = (29, 1) # Canto inferior esquerdo
            elif player.col <= 15 and player.row > 15: self.target = (1, 28) # Canto superior direito
            else: self.target = (29, 28) # Canto inferior direito
        else:
            self.image = self.assets[self.ghost_type]
            # Se estiver na caixa, o alvo é a saída. Senão, usa a IA.
            if self.is_in_box:
                self.target = (11, 14) # Posição logo acima da porta
            else:
                self.target = self.get_ai_target(player, blinky)

        # 2. Mover o fantasma um passo
        self.move_one_step(level_board)

        # 3. Atualizar a posição em pixels e o estado da caixa
        self.rect.center = self.get_pixel_pos()
        self.is_in_box = 13 <= self.row <= 15 and 12 <= self.col <= 16

    def move_one_step(self, level_board):
        """ Esta função reimplementa a lógica de decisão original, mas em um grid. """
        valid_moves = self.get_valid_moves(level_board)
        if not valid_moves:
            return

        # Priorizar movimento horizontal ou vertical com base na posição do alvo
        # Se o alvo está mais longe na horizontal
        if abs(self.target[1] - self.col) > abs(self.target[0] - self.row):
            # Tenta mover na direção horizontal preferida
            pref_dir = 0 if self.target[1] > self.col else 1 # 0: Direita, 1: Esquerda
            if pref_dir in valid_moves:
                self.direction = pref_dir
            # Se não puder, tenta mover na vertical
            elif 2 in valid_moves: self.direction = 2
            elif 3 in valid_moves: self.direction = 3
            # Se preso, usa o movimento restante (reverter)
            else: self.direction = list(valid_moves.keys())[0]
        # Se o alvo está mais longe na vertical (ou mesma distância)
        else:
            # Tenta mover na direção vertical preferida
            pref_dir = 3 if self.target[0] > self.row else 2 # 3: Baixo, 2: Cima
            if pref_dir in valid_moves:
                self.direction = pref_dir
            # Se não puder, tenta mover na horizontal
            elif 0 in valid_moves: self.direction = 0
            elif 1 in valid_moves: self.direction = 1
            # Se preso, usa o movimento restante (reverter)
            else: self.direction = list(valid_moves.keys())[0]

        self.row, self.col = valid_moves[self.direction]

    def get_valid_moves(self, level_board):
        moves = {}
        # Direita (0)
        if self.col < 29 and (level_board[self.row][self.col + 1] < 3 or (self.is_in_box and level_board[self.row][self.col + 1] == 9)):
            moves[0] = (self.row, self.col + 1)
        # Esquerda (1)
        if self.col > 0 and (level_board[self.row][self.col - 1] < 3 or (self.is_in_box and level_board[self.row][self.col - 1] == 9)):
            moves[1] = (self.row, self.col - 1)
        # Cima (2)
        if self.row > 0 and (level_board[self.row - 1][self.col] < 3 or (self.is_in_box and level_board[self.row - 1][self.col] == 9)):
            moves[2] = (self.row - 1, self.col)
        # Baixo (3)
        if self.row < 30 and (level_board[self.row + 1][self.col] < 3 or (self.is_in_box and level_board[self.row + 1][self.col] == 9)):
            moves[3] = (self.row + 1, self.col)

        # Impede que o fantasma reverta a direção, a menos que seja a única opção
        if len(moves) > 1 and not self.is_in_box:
            opposite_dir = {0: 1, 1: 0, 2: 3, 3: 2}.get(self.direction)
            if opposite_dir in moves:
                del moves[opposite_dir]
        
        return moves

    def get_ai_target(self, player, blinky):
        # A lógica de IA para definir o alvo permanece a mesma
        if self.ghost_type == 'blinky':
            return (player.row, player.col)
        if self.ghost_type == 'pinky':
            r, c = player.row, player.col
            if player.direction == 0: c = min(28, c + 4)
            elif player.direction == 1: c = max(1, c - 4)
            elif player.direction == 2: r = max(1, r - 4)
            elif player.direction == 3: r = min(29, r + 4)
            return (r, c)
        if self.ghost_type == 'inky':
             r, c = player.row, player.col
             if player.direction == 0: c = min(28, c + 2)
             elif player.direction == 1: c = max(1, c - 2)
             elif player.direction == 2: r = max(1, r - 2)
             elif player.direction == 3: r = min(29, r + 2)
             return (blinky.row + (r - blinky.row) * 2, blinky.col + (c - blinky.col) * 2)
        if self.ghost_type == 'clyde':
            dist = math.hypot(self.row - player.row, self.col - player.col)
            if dist > 8:
                return (player.row, player.col)
            else: 
                return (29, 1)
        return (player.row, player.col)
    
    def handle_teleport(self):
        if self.col < 0:
            self.col = 29
        elif self.col > 29:
            self.col = 0

    def get_pixel_pos(self):
        return (self.col * TILE_WIDTH + TILE_WIDTH // 2, self.row * TILE_HEIGHT + TILE_HEIGHT // 2)

    def reset(self):
        self.row, self.col = self.home_pos
        self.is_dead = False
        self.is_in_box = True
        self.direction = 0

class Pellet(pygame.sprite.Sprite):
    def __init__(self, position, pellet_type):
        super().__init__()
        self.type = pellet_type
        if self.type == 1: # Normal
            self.image = pygame.Surface([4, 4])
            self.image.fill('white')
        else: # Power
            self.image = pygame.Surface([8, 8])
            self.image.fill('white')
        self.rect = self.image.get_rect(center=position)

# --- Funções do Jogo ---
def draw_board(screen, level):
    # Função para desenhar o labirinto (simplificada, pois as pastilhas agora são sprites)
    color = 'blue'
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] == 3: pygame.draw.line(screen, color, (j * TILE_WIDTH + (0.5 * TILE_WIDTH), i * TILE_HEIGHT), (j * TILE_WIDTH + (0.5 * TILE_WIDTH), i * TILE_HEIGHT + TILE_HEIGHT), 3)
            if level[i][j] == 4: pygame.draw.line(screen, color, (j * TILE_WIDTH, i * TILE_HEIGHT + (0.5 * TILE_HEIGHT)), (j * TILE_WIDTH + TILE_WIDTH, i * TILE_HEIGHT + (0.5 * TILE_HEIGHT)), 3)
            # ... (adicionar outros elementos do mapa se necessário)

def draw_info(screen, font, score, lives, assets):
    score_text = font.render(f'Score: {score}', True, 'white')
    screen.blit(score_text, (10, HEIGHT - 40))
    for i in range(lives):
        screen.blit(assets['player'][0], (WIDTH - 150 + i * 40, HEIGHT - 40))

# --- DRL - Lógica do Agente ---
device = torch.device("cpu") # Usando CPU para este modelo

# Hiperparâmetros
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 3000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LR = 3e-4
model_file = 'dql_discrete_model.pth'

#parametros de reward:
score_multiplier = 5

# A arquitetura da rede não muda, mas o número de observações sim
N_ACTIONS = 4
N_OBSERVATIONS = 19 # (pac_r, pac_c) + 4*(ghost_r, ghost_c, is_vuln) + (powerup_time) + 4*(wall_dir) = 2+12+1+4 = 19

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory: # (código da ReplayMemory permanece o mesmo)
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class DQN(nn.Module): # (código da DQN permanece o mesmo)
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def get_state_discrete(player, ghosts, powerup_counter, level_board):
    features = []
    
    # Normalizadores
    max_row, max_col = 30, 29

    # Posição do jogador
    features.append(player.row / max_row)
    features.append(player.col / max_col)
    
    # Posição e estado dos fantasmas
    for ghost in sorted(ghosts, key=lambda g: g.id): # Ordenar para consistência
        features.append(ghost.row / max_row)
        features.append(ghost.col / max_col)
        features.append(1.0 if not ghost.is_dead and powerup_counter > 0 else 0.0) # É vulnerável?
    
    # Tempo de powerup
    features.append(powerup_counter / 600.0)
    
    # Visão de paredes
    valid_moves = get_valid_moves_for_pos(player.row, player.col, level_board)
    features.append(1.0 if 0 in valid_moves else 0.0) # Pode ir para Direita?
    features.append(1.0 if 1 in valid_moves else 0.0) # Esquerda?
    features.append(1.0 if 2 in valid_moves else 0.0) # Cima?
    features.append(1.0 if 3 in valid_moves else 0.0) # Baixo?

    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

def get_valid_moves_for_pos(row, col, level_board):
    moves = {}
    if col < 29 and level_board[row][col + 1] < 3: moves[0] = (row, col + 1)
    if col > 0 and level_board[row][col - 1] < 3: moves[1] = (row, col - 1)
    if row > 0 and level_board[row - 1][col] < 3: moves[2] = (row - 1, col)
    if row < 30 and level_board[row + 1][col] < 3: moves[3] = (row + 1, col)
    return moves

def choose_action(state, valid_moves, policy_net, steps_done, current_direction):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    if random.random() > eps_threshold:  # Exploitation
        with torch.no_grad():
            q_values = policy_net(state)
            
            # --- NOVA LÓGICA DE INÉRCIA/MOMENTUM ---
            # Adiciona um pequeno bônus ao Q-value da direção atual para incentivar a continuação.
            # Isso ajuda a quebrar empates e a sair de cantos.
            if current_direction in valid_moves:
                q_values[0][current_direction] += 0.2 # O valor do bônus pode ser ajustado

            # Mascarar ações fisicamente inválidas (bater na parede)
            for i in range(N_ACTIONS):
                if i not in valid_moves:
                    q_values[0][i] = -float('inf')
            
            return torch.tensor([[q_values.argmax().item()]], device=device, dtype=torch.long)
    else:  # Exploration
        action = random.choice(list(valid_moves.keys()))
        return torch.tensor([[action]], device=device, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer): # (código de otimização permanece o mesmo)
    if len(memory) < BATCH_SIZE: return
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

# --- Função Principal do Jogo ---
def main():
    from board import boards # Importa os dados do tabuleiro
    
    pygame.init()
    assets = load_assets()
    clock = pygame.time.Clock()

    # --- Inicialização do Agente DRL ---
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

    # --- Configuração de Runs ---
    nTreino = int(input("Quantas runs de treino o programa deve fazer?: "))
    nRender = int(input("Quantas runs você deseja assistir após o treino?: "))

    history_len = 50 # Rastrear os últimos 50 turnos
    stagnation_tracker = deque(maxlen=history_len)
    
    for episode in range(nTreino + nRender):
        # --- Reset do Ambiente a cada Episódio ---
        level = copy.deepcopy(boards)
        
        player = Player(assets, initial_pos=(23, 14))
        blinky = Ghost(assets, initial_pos=(4, 2), ghost_type='blinky', ghost_id=0)
        pinky = Ghost(assets, initial_pos=(13, 12), ghost_type='pinky', ghost_id=1)
        inky = Ghost(assets, initial_pos=(13, 14), ghost_type='inky', ghost_id=2)
        clyde = Ghost(assets, initial_pos=(13, 16), ghost_type='clyde', ghost_id=3)
        
        all_sprites = pygame.sprite.Group(player, blinky, pinky, inky, clyde)
        ghosts = pygame.sprite.Group(blinky, pinky, inky, clyde)
        
        pellets = pygame.sprite.Group()
        for r_idx, row in enumerate(level):
            for c_idx, val in enumerate(row):
                if val == 1 or val == 2:
                    pos = (c_idx * TILE_WIDTH + TILE_WIDTH // 2, r_idx * TILE_HEIGHT + TILE_HEIGHT // 2)
                    pellets.add(Pellet(pos, val))

        score = 0
        powerup_counter = 0
        eaten_ghost_count = 0

        last_dist_to_ghost = None
        last_dist_to_vulnerable = None
        
        render_this_run = (episode >= nTreino)

        # Loop de um episódio (baseado em turnos)
        for turn in range(2000): # Limite de turnos para evitar loops infinitos

            stagnation_tracker.append({'pos': (player.row, player.col), 'score': score})

            # --- Processar Eventos do Pygame (ESSENCIAL PARA EVITAR CONGELAMENTO) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Salva o progresso e sai de forma limpa se a janela for fechada
                    torch.save(policy_net.state_dict(), model_file)
                    print("Jogo interrompido pelo usuário. Modelo salvo.")
                    pygame.quit()
                    # Usamos return para sair da função main e terminar o programa completamente
                    return 
            # --- FIM DO BLOCO DE EVENTOS ---

            # --- Ciclo de Decisão e Ação ---
            valid_moves = get_valid_moves_for_pos(player.row, player.col, level)
            if not valid_moves: break # Fim de jogo se o jogador ficar preso
            
            state = get_state_discrete(player, ghosts, powerup_counter, level)
            action_tensor = choose_action(state, valid_moves, policy_net, steps_done, player.direction)
            action = action_tensor.item()
            steps_done += 1
            
            # --- Atualização do Mundo ---
            score_before = score
            player.update(action, level)
            player.handle_teleport()
            ghosts.update(player, blinky, powerup_counter > 0, level)
            # Verificação de teletransporte para cada fantasma
            for ghost in ghosts:
                ghost.handle_teleport()

            # --- Verificação de Consequências e Recompensa ---
            reward = 0
            
            # Colisão com pastilhas
            collided_pellets = pygame.sprite.spritecollide(player, pellets, True) # True remove a pastilha
            for pellet in collided_pellets:
                if pellet.type == 1:
                    score += 10
                else: # Power pellet
                    score += 50
                    powerup_counter = 600 # Ativa o power-up
                    eaten_ghost_count = 0

            # Colisão com fantasmas
            collided_ghosts = pygame.sprite.spritecollide(player, ghosts, False)
            if collided_ghosts:
                for ghost in collided_ghosts:
                    if powerup_counter > 0 and not ghost.is_dead:
                        ghost.is_dead = True
                        eaten_ghost_count += 1
                        score += (2 ** eaten_ghost_count) * 100
                    elif not ghost.is_dead:
                        player.lives -= 1
                        reward -= 500 # Grande penalidade
                        # Reset de posições para a mesma vida
                        player.row, player.col = (23, 14)
                        for g in ghosts: g.reset() # Resetar fantasmas
                        break # Termina o turno aqui

            ### CÁLCULO DA RECOMPENSA (REWARD SHAPING) ###
            
            # 1. Recompensa base por pontuação (eventos discretos)
            reward = score_multiplier * (score - score_before)

            # 2. Grandes penalidades e bônus por eventos de fim de jogo/vida
            if player.lives < 3 and score_before == score: # Uma forma de detectar se a vida foi perdida neste turno
                 # (Esta lógica pode precisar de ajuste, a forma mais simples é ter uma flag)
                 # Vamos assumir uma flag `player_lost_life_this_turn` que é setada como True no bloco de colisão
                 # if player_lost_life_this_turn:
                 #    reward -= 500
                 pass # Manteremos a sua lógica original por enquanto para evitar mais bugs

            # Se o jogador morreu (última vida)
            if player.lives == 0:
                reward -= 500

            # Se o jogador venceu
            if len(pellets) == 0:
                reward += 1000

            # 3. Modelagem de comportamento (incentivos/desincentivos contínuos)
            
            # 3a. Penalidade por Estagnação
            is_stagnant = False
            if len(stagnation_tracker) == history_len:
                score_has_changed = any(hist['score'] != score for hist in stagnation_tracker)
                start_pos = stagnation_tracker[0]['pos']
                manhattan_dist = abs(player.row - start_pos[0]) + abs(player.col - start_pos[1])
                if not score_has_changed and manhattan_dist < 4:
                    is_stagnant = True
            
            if is_stagnant:
                reward -= 10

            # 3b. Incentivo Dinâmico para Interagir com Fantasmas (Fugir ou Caçar)
            if powerup_counter <= 0: # CENÁRIO 1: FUGIR DE FANTASMAS PERIGOSOS
                dist_to_dangerous_ghost = float('inf')
                for ghost in ghosts:
                    if not ghost.is_dead:
                        dist = math.hypot(player.row - ghost.row, player.col - ghost.col)
                        dist_to_dangerous_ghost = min(dist_to_dangerous_ghost, dist)
                
                # Recompensa a MUDANÇA na distância para incentivar o AFASTAMENTO
                if last_dist_to_ghost is not None and dist_to_dangerous_ghost != float('inf'):
                    reward_change = dist_to_dangerous_ghost - last_dist_to_ghost
                    reward += 0.5 * reward_change # Recompensa positiva se a distância aumenta
                
                # Atualiza a variável para o próximo turno
                last_dist_to_ghost = dist_to_dangerous_ghost if dist_to_dangerous_ghost != float('inf') else last_dist_to_ghost
                last_dist_to_vulnerable = None # Reseta a outra variável
            
            else: # CENÁRIO 2: CAÇAR FANTASMAS VULNERÁVEIS
                dist_to_vulnerable_ghost = float('inf')
                for ghost in ghosts:
                    if not ghost.is_dead: # Todos os não mortos são vulneráveis
                        dist = math.hypot(player.row - ghost.row, player.col - ghost.col)
                        dist_to_vulnerable_ghost = min(dist_to_vulnerable_ghost, dist)
                
                # Recompensa a MUDANÇA na distância para incentivar a APROXIMAÇÃO
                if last_dist_to_vulnerable is not None and dist_to_vulnerable_ghost != float('inf'):
                    # O sinal é invertido: recompensa se a distância diminui
                    reward_change = last_dist_to_vulnerable - dist_to_vulnerable_ghost
                    reward += 0.7 * reward_change # Recompensa um pouco mais a caça do que a fuga

                # Atualiza a variável para o próximo turno
                last_dist_to_vulnerable = dist_to_vulnerable_ghost if dist_to_vulnerable_ghost != float('inf') else last_dist_to_vulnerable
                last_dist_to_ghost = None # Reseta a outra variável
            
            # 4. Custo de vida (penalidade de tempo)
            reward -= 1
            ### FIM DO CÁLCULO DA RECOMPENSA ###

            # --- DRL - Aprendizado ---
            done = (player.lives <= 0) or (len(pellets) == 0)
            
            if done:
                next_state = None
                if len(pellets) == 0: reward += 1000 # Grande recompensa por vencer
            else:
                next_state = get_state_discrete(player, ghosts, powerup_counter, level)

            memory.push(state, action_tensor, next_state, torch.tensor([reward], device=device))
            
            if steps_done % 4 == 0:
                optimize_model(memory, policy_net, target_net, optimizer)
            
            # Atualiza contador de powerup
            if powerup_counter > 0: powerup_counter -= 15 # Reduz baseado em turnos
            
            # Fantasmas renascem
            for ghost in ghosts:
                if ghost.is_dead and ghost.row == 13 and ghost.col == 14:
                    ghost.is_dead = False
            
            # --- Renderização (se aplicável) ---
            if render_this_run:
                SCREEN.fill('black')
                draw_board(SCREEN, level)
                pellets.draw(SCREEN)
                all_sprites.draw(SCREEN)
                draw_info(SCREEN, assets['font'], score, player.lives, assets)
                pygame.display.flip()
                clock.tick(15) # Controla a velocidade da animação

            if done:
                break
        
        print(f"Episódio {episode + 1} concluído. Score: {score}. Passos: {turn + 1}")

        # Atualiza a rede alvo periodicamente
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    # Salva o modelo final
    torch.save(policy_net.state_dict(), model_file)
    print("Treinamento concluído. Modelo salvo.")
    pygame.quit()


if __name__ == '__main__':
    main()