import numpy as np
import random

class RandomAgent:
    """
    Agent qui choisit des actions aléatoires
    - Baseline sans apprentissage
    """
    def __init__(self, env):
        self.env = env
        self.name = "Agent Aléatoire"
    
    def choose_action(self, state, epsilon=0):
        """Choisit une action aléatoire"""
        return random.choice(self.env.actions)
    
    def learn(self, *args):
        """Pas d'apprentissage"""
        pass
    
    def get_q_values(self):
        """Pas de Q-values pour cet agent"""
        return None
    
    def get_q_grid(self):
        """Retourne une grille vide"""
        return np.zeros((self.env.size, self.env.size))

class ValueIterationAgent:
    """
    Agent utilisant Value Iteration
    - Apprentissage hors-ligne
    - Planification dynamique
    """
    def __init__(self, env, gamma=0.9, theta=1e-3):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = {}
        self.name = "Agent Value Iteration"
        
        # Donner une valeur élevée au but
        self.env.state_values[self.env.goal_pos[0], self.env.goal_pos[1]] = 100
        
        # Apprentissage
        self.value_iteration()
        self.extract_policy()
    
    def value_iteration(self):
        """Algorithme Value Iteration corrigé"""
        states = self.env.get_all_states()
        
        iteration = 0
        while True:
            delta = 0
            
            for state_idx in states:
                x, y = self.env.get_coords(state_idx)
                
                # Ne pas mettre à jour la valeur du but
                if [x, y] == self.env.goal_pos:
                    continue
                
                # Ne pas mettre à jour les obstacles
                if [x, y] in self.env.obstacles:
                    continue
                
                old_value = self.env.state_values[x, y]
                best_value = -np.inf
                
                # Pour chaque action possible
                for action in self.env.actions:
                    dx, dy = self.env.action_vectors[action]
                    new_x, new_y = x + dx, y + dy
                    
                    # Vérifier si le mouvement est valide
                    if not (0 <= new_x < self.env.size and 0 <= new_y < self.env.size):
                        new_x, new_y = x, y  # Rester sur place
                    
                    if [new_x, new_y] in self.env.obstacles:
                        new_x, new_y = x, y  # Rester sur place
                    
                    # Calculer la récompense
                    if [new_x, new_y] == self.env.goal_pos:
                        reward = 100.0
                    else:
                        # Calculer la distance au but
                        distance = abs(new_x - self.env.goal_pos[0]) + abs(new_y - self.env.goal_pos[1])
                        max_distance = (self.env.size - 1) * 2
                        
                        # Récompense basée sur la distance
                        if distance == 1:
                            reward = 5.0
                        elif distance <= 3:
                            reward = 1.0
                        else:
                            reward = -0.1
                    
                    # Calculer la valeur
                    value = reward + self.gamma * self.env.state_values[new_x, new_y]
                    
                    if value > best_value:
                        best_value = value
                
                # Mettre à jour la valeur de l'état
                self.env.state_values[x, y] = best_value
                delta = max(delta, abs(old_value - best_value))
            
            iteration += 1
            if delta < self.theta:
                break
        
        print(f"Value Iteration convergée en {iteration} itérations")
    
    def extract_policy(self):
        """Extrait la politique optimale"""
        states = self.env.get_all_states()
        
        for state_idx in states:
            x, y = self.env.get_coords(state_idx)
            
            # Si c'est le but, pas d'action
            if [x, y] == self.env.goal_pos:
                self.policy[state_idx] = None
                continue
            
            # Si c'est un obstacle, pas d'action
            if [x, y] in self.env.obstacles:
                self.policy[state_idx] = None
                continue
            
            best_action = None
            best_value = -np.inf
            
            # Pour chaque action
            for action in self.env.actions:
                dx, dy = self.env.action_vectors[action]
                new_x, new_y = x + dx, y + dy
                
                # Vérifier si le mouvement est valide
                if not (0 <= new_x < self.env.size and 0 <= new_y < self.env.size):
                    continue
                
                if [new_x, new_y] in self.env.obstacles:
                    continue
                
                # Vérifier la valeur de l'état suivant
                value = self.env.state_values[new_x, new_y]
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            self.policy[state_idx] = best_action
    
    def choose_action(self, state, epsilon=0):
        """Choisit une action selon la politique optimale"""
        return self.policy.get(state, random.choice(self.env.actions))
    
    def learn(self, *args):
        """Pas d'apprentissage en ligne"""
        pass
    
    def get_q_values(self):
        """Pas de Q-table pour cet agent"""
        return None
    
    def get_q_grid(self):
        """Retourne les valeurs d'état sous forme de grille"""
        return self.env.state_values.copy()

class QLearningAgent:
    """
    Agent utilisant Q-Learning pour apprendre
    - Exploration vs exploitation (epsilon-greedy)
    - Apprentissage par renforcement
    - Adaptatif au déplacement du but
    """
    def __init__(self, env, alpha=0.3, gamma=0.9, epsilon=0.8):
        self.env = env
        self.alpha = alpha      # Taux d'apprentissage
        self.gamma = gamma      # Facteur d'actualisation
        self.epsilon = epsilon  # Taux d'exploration initial
        self.initial_epsilon = epsilon
        
        # Table Q: [état, action] -> valeur
        self.q_table = np.zeros((env.n_states, env.n_actions))
        
        # Statistiques
        self.learning_history = []
        self.name = "Agent Q-Learning"
    
    def choose_action(self, state, epsilon=None):
        """
        Choisit une action selon stratégie epsilon-greedy
        - Exploration: action aléatoire (epsilon)
        - Exploitation: meilleure action selon Q-table (1-epsilon)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Exploration: action aléatoire
        if random.random() < epsilon:
            return random.choice(self.env.actions)
        
        # Exploitation: meilleure action
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            
            # Gestion des égalités
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Met à jour la Q-table selon l'algorithme Q-Learning:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
        """
        # Valeur actuelle
        current_q = self.q_table[state, action]
        
        if done:
            target = reward  # Pas de futur pour les états terminaux
        else:
            # Meilleure valeur future
            future_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * future_q
        
        # Mise à jour
        self.q_table[state, action] += self.alpha * (target - current_q)
    
    def update_epsilon(self, episode, total_episodes):
        """Réduit epsilon progressivement (exploration → exploitation)"""
        self.epsilon = max(0.01, self.initial_epsilon * (1 - episode / total_episodes))
        return self.epsilon
    
    def get_q_values(self):
        """Retourne la Q-table"""
        return self.q_table
    
    def get_q_grid(self):
        """Convertit la Q-table en grille 2D pour visualisation"""
        q_grid = np.zeros((self.env.size, self.env.size))
        for state in range(self.env.n_states):
            x, y = self.env.get_coords(state)
            q_grid[x, y] = np.max(self.q_table[state])
        return q_grid
    
    def reset_learning(self):
        """Réinitialise l'apprentissage"""
        self.q_table = np.zeros((self.env.n_states, self.env.n_actions))
        self.epsilon = self.initial_epsilon
        print("Apprentissage réinitialisé")