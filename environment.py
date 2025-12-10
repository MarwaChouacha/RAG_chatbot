import numpy as np

class GridWorldEnv:
    """
    Environnement GridWorld pour Reinforcement Learning
    - Agent, but, obstacles
    - États discrets
    - Actions: haut, bas, gauche, droite
    - Récompenses adaptatives
    """
    def __init__(self, size=5):
        self.size = size
        
        # Positions initiales
        self.agent_pos = [0, 0]
        self.goal_pos = [size-1, size-1]
        self.obstacles = [[1, 2], [3, 3]]
        
        # Actions possibles
        self.actions = [0, 1, 2, 3]  # 0:haut, 1:bas, 2:gauche, 3:droite
        self.action_names = ['Haut', 'Bas', 'Gauche', 'Droite']
        self.action_vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Espace d'états
        self.n_states = size * size
        self.n_actions = len(self.actions)
        
        # Valeurs initiales
        self.state_values = np.zeros((size, size))
        self.visited = np.zeros((size, size))
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.agent_pos = [0, 0]
        self.visited = np.zeros((self.size, self.size))
        return self.get_state()
    
    def get_state(self):
        """Retourne l'état courant sous forme d'index"""
        x, y = self.agent_pos
        return x * self.size + y
    
    def get_coords(self, state):
        """Convertit un index d'état en coordonnées (x, y)"""
        x = state // self.size
        y = state % self.size
        return x, y
    
    def step(self, action):
        """
        Exécute une action dans l'environnement
        Retourne: (next_state, reward, done)
        """
        x, y = self.agent_pos
        dx, dy = self.action_vectors[action]
        new_x, new_y = x + dx, y + dy
        
        # Vérifier les limites et obstacles
        if self._is_valid_position(new_x, new_y):
            self.agent_pos = [new_x, new_y]
        
        # Marquer comme visité
        self.visited[self.agent_pos[0], self.agent_pos[1]] += 1
        
        # Calculer la récompense
        reward, done = self._calculate_reward()
        
        return self.get_state(), reward, done
    
    def _is_valid_position(self, x, y):
        """Vérifie si une position est valide"""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if [x, y] in self.obstacles:
            return False
        return True
    
    def _calculate_reward(self):
        """Calcule la récompense"""
        # Récompense pour atteindre le but
        if self.agent_pos == self.goal_pos:
            return 100.0, True
        
        # Pénalité pour obstacle
        elif self.agent_pos in self.obstacles:
            return -10.0, True
        
        # Récompense pour se rapprocher du but
        else:
            # Distance de Manhattan au but
            distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            
            # Récompense basée sur la distance
            if distance == 1:  # Juste à côté du but
                return 5.0, False
            elif distance <= 3:  # Proche du but
                return 1.0, False
            else:
                return -0.1, False
    
    def move_goal(self, new_position):
        """Déplace le but à une nouvelle position"""
        if self._is_valid_position(new_position[0], new_position[1]):
            self.goal_pos = new_position
            print(f"But déplacé à {new_position}")
            return True
        else:
            print(f"Position {new_position} invalide pour le but")
            return False
    
    def get_all_states(self):
        """Retourne tous les états valides"""
        states = []
        for i in range(self.size):
            for j in range(self.size):
                if [i, j] not in self.obstacles:
                    states.append(i * self.size + j)
        return states
    
    def render(self):
        """Affiche l'environnement en console"""
        grid = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if [i, j] == self.agent_pos:
                    row.append('A')
                elif [i, j] == self.goal_pos:
                    row.append('G')
                elif [i, j] in self.obstacles:
                    row.append('X')
                else:
                    visits = self.visited[i, j]
                    if visits > 0:
                        row.append(str(min(9, int(visits))))
                    else:
                        row.append('.')
            grid.append(' '.join(row))
        return '\n'.join(grid)