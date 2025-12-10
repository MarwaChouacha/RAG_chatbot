import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import time
import sys

# Import des modules locaux
from environment import GridWorldEnv
from agents import RandomAgent, ValueIterationAgent, QLearningAgent

class InteractiveGrid:
    """
    Interface interactive pour GridWorld
    - Affichage en temps réel
    - Déplacement pas à pas
    - Contrôles visuels
    """
    
    def __init__(self, env, agent, title="GridWorld Interactive"):
        self.env = env
        self.agent = agent
        self.title = title
        self.fig = None
        self.ax = None
        self.steps = 0
        self.total_reward = 0
        self.running = True
        self.auto_mode = False
        self.last_action = None
        
        # Initialiser l'affichage
        self.init_display()
    
    def init_display(self):
        """Initialise l'affichage graphique"""
        plt.ion()  # Mode interactif
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Configurer l'affichage
        self.ax.set_xlim(-0.5, self.env.size - 0.5)
        self.ax.set_ylim(-0.5, self.env.size - 0.5)
        self.ax.set_xticks(np.arange(self.env.size))
        self.ax.set_yticks(np.arange(self.env.size))
        self.ax.set_xticklabels(np.arange(self.env.size))
        self.ax.set_yticklabels(np.arange(self.env.size))
        self.ax.grid(True, which='both', color='black', linestyle='-', linewidth=1, alpha=0.3)
        self.ax.set_xlabel('Colonne')
        self.ax.set_ylabel('Ligne')
        self.ax.set_aspect('equal')
        
        # Stocker les éléments graphiques
        self.elements = {
            'obstacles': [],
            'goal': None,
            'agent': None,
            'texts': {},
            'path': []
        }
        
        # Dessiner l'environnement initial
        self.draw_environment()
        
        # Titre
        self.fig.suptitle(f"{self.title}\nCommandes: 'n'=pas suivant, 'a'=auto, 'r'=reset, 'q'=quitter", 
                         fontsize=12, fontweight='bold')
        
        # Connecter les événements clavier
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
    
    def draw_environment(self):
        """Dessine tous les éléments de l'environnement"""
        self.ax.clear()
        self.ax.grid(True, which='both', color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Dessiner les obstacles
        for obs in self.env.obstacles:
            rect = patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1,
                                   facecolor='red', alpha=0.5, edgecolor='darkred')
            self.ax.add_patch(rect)
            self.ax.text(obs[1], obs[0], 'X', ha='center', va='center',
                       fontweight='bold', fontsize=14, color='white')
        
        # Dessiner le but
        goal_rect = patches.Rectangle((self.env.goal_pos[1]-0.5, self.env.goal_pos[0]-0.5), 1, 1,
                                    facecolor='green', alpha=0.7, edgecolor='darkgreen')
        self.ax.add_patch(goal_rect)
        self.ax.text(self.env.goal_pos[1], self.env.goal_pos[0], 'G', ha='center', va='center',
                   fontweight='bold', fontsize=16, color='white')
        
        # Dessiner les valeurs d'état (pour Value Iteration) ou Q-values (pour Q-Learning)
        if hasattr(self.agent, 'get_q_grid'):
            q_grid = self.agent.get_q_grid()
            for i in range(self.env.size):
                for j in range(self.env.size):
                    if [i, j] not in self.env.obstacles and [i, j] != self.env.goal_pos:
                        q_val = q_grid[i, j]
                        if q_val != 0:
                            # Couleur basée sur la valeur
                            if q_val > 0:
                                color = 'lightgreen'
                                alpha = min(0.3 + q_val / 20, 0.7)
                            else:
                                color = 'lightcoral'
                                alpha = min(0.3 + abs(q_val) / 10, 0.7)
                            
                            rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                                   facecolor=color, alpha=alpha, edgecolor='gray')
                            self.ax.add_patch(rect)
                            if abs(q_val) > 0.1:  # N'afficher que les valeurs significatives
                                self.ax.text(j, i, f'{q_val:.1f}', ha='center', va='center',
                                           fontsize=8, color='black' if abs(q_val) < 5 else 'white')
        
        # Dessiner le chemin parcouru
        for i, (x, y) in enumerate(self.elements['path']):
            # Taille décroissante pour les anciens points
            size = max(0.1, 0.2 * (1 - i / max(1, len(self.elements['path']))))
            circle = patches.Circle((y, x), radius=size, color='blue', alpha=0.3)
            self.ax.add_patch(circle)
        
        # Dessiner l'agent
        agent_circle = patches.Circle((self.env.agent_pos[1], self.env.agent_pos[0]), 
                                    radius=0.4, facecolor='blue', edgecolor='darkblue', linewidth=2)
        self.ax.add_patch(agent_circle)
        self.ax.text(self.env.agent_pos[1], self.env.agent_pos[0], 'A', ha='center', va='center',
                   fontweight='bold', fontsize=16, color='white')
        
        # Information en bas
        info_text = f"Pas: {self.steps} | Récompense: {self.total_reward:.2f} | Position: {self.env.agent_pos}"
        if hasattr(self.agent, 'epsilon'):
            info_text += f" | ε: {self.agent.epsilon:.3f}"
        
        self.ax.text(0.5, -0.08, info_text, ha='center', va='center', transform=self.ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Afficher la dernière action
        if self.last_action is not None:
            action_text = f"Dernière action: {self.env.action_names[self.last_action]}"
            self.ax.text(0.5, -0.13, action_text, ha='center', va='center', transform=self.ax.transAxes,
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
        
        # Configuration
        self.ax.set_xlim(-0.5, self.env.size - 0.5)
        self.ax.set_ylim(-0.5, self.env.size - 0.5)
        self.ax.set_xticks(np.arange(self.env.size))
        self.ax.set_yticks(np.arange(self.env.size))
        self.ax.set_xticklabels(np.arange(self.env.size))
        self.ax.set_yticklabels(np.arange(self.env.size))
        self.ax.set_xlabel('Colonne')
        self.ax.set_ylabel('Ligne')
        self.ax.set_title(f"{self.agent.name}", fontsize=12, fontweight='bold')
    
    def take_step(self):
        """Prend un pas dans l'environnement"""
        # Sauvegarder la position actuelle pour le chemin
        self.elements['path'].append(tuple(self.env.agent_pos))
        
        # Obtenir l'état actuel
        state = self.env.get_state()
        
        # Choisir une action
        if hasattr(self.agent, 'epsilon'):
            # Pour Q-Learning, utiliser l'epsilon actuel
            action = self.agent.choose_action(state)
        else:
            # Pour Value Iteration ou Random, pas d'epsilon
            action = self.agent.choose_action(state)
        
        # Enregistrer la dernière action
        self.last_action = action
        
        # Exécuter l'action
        next_state, reward, done = self.env.step(action)
        
        # Apprendre (si l'agent le supporte)
        if hasattr(self.agent, 'learn'):
            self.agent.learn(state, action, reward, next_state, done)
        
        # Mettre à jour les statistiques
        self.steps += 1
        self.total_reward += reward
        
        # Afficher dans la console
        print(f"  Pas {self.steps}: {self.env.action_names[action]} → Position {self.env.agent_pos}, Récompense: {reward:.2f}")
        
        # Redessiner
        self.draw_environment()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Vérifier si terminé
        if done:
            if reward > 0:
                print(f"\n✓ OBJECTIF ATTEINT! En {self.steps} pas, Récompense totale: {self.total_reward:.2f}")
                print("🎉 SUCCÈS: L'agent a trouvé le but!")
            else:
                print(f"\n💥 ÉCHEC! En {self.steps} pas")
                print("💥 L'agent a touché un obstacle!")
            
            self.running = False
            return True
        
        return False
    
    def run_until_goal(self, max_steps=100, delay=0.3):
        """Exécute jusqu'à atteindre le but ou max_steps"""
        self.steps = 0
        self.total_reward = 0
        self.elements['path'] = []
        self.last_action = None
        
        print(f"\nDémarrage de la séquence (max {max_steps} pas)...")
        
        for step in range(max_steps):
            done = self.take_step()
            
            if done:
                print(f"\n🎉 Objectif atteint en {step+1} pas!")
                break
            
            if delay > 0:
                time.sleep(delay)
        
        if not done:
            print(f"\n⏰ Maximum de pas atteint ({max_steps})")
        
        return done
    
    def on_key_press(self, event):
        """Gère les événements clavier"""
        if event.key == 'n':
            # Pas suivant
            if self.running:
                self.take_step()
            else:
                print("La séquence est terminée. Appuyez sur 'r' pour réinitialiser.")
        
        elif event.key == 'a':
            # Exécuter automatiquement jusqu'au but
            if not self.auto_mode:
                self.auto_mode = True
                print("\nMode automatique activé...")
                self.run_until_goal(max_steps=50, delay=0.1)
                self.auto_mode = False
        
        elif event.key == 'r':
            # Réinitialiser
            self.env.reset()
            self.steps = 0
            self.total_reward = 0
            self.elements['path'] = []
            self.last_action = None
            self.running = True
            
            print("\nEnvironnement réinitialisé!")
            self.draw_environment()
            self.fig.canvas.draw()
        
        elif event.key == 'q':
            # Quitter
            print("\nFermeture de l'interface...")
            self.running = False
            plt.close()
        
        elif event.key == 's':
            # Afficher les statistiques
            print("\n📊 STATISTIQUES:")
            print(f"  Pas effectués: {self.steps}")
            print(f"  Récompense totale: {self.total_reward:.2f}")
            print(f"  Position actuelle: {self.env.agent_pos}")
            print(f"  But: {self.env.goal_pos}")
    
    def on_close(self, event):
        """Gère la fermeture de la fenêtre"""
        self.running = False
    
    def interactive_mode(self):
        """Mode interactif contrôlé par l'utilisateur"""
        print("\n" + "="*60)
        print("MODE INTERACTIF")
        print("="*60)
        print("Commandes:")
        print("  n - Pas suivant")
        print("  a - Exécuter automatiquement jusqu'au but")
        print("  r - Réinitialiser")
        print("  s - Afficher les statistiques")
        print("  q - Quitter")
        print("\nPosition initiale:", self.env.agent_pos)
        print("But:", self.env.goal_pos)
        print("Obstacles:", self.env.obstacles)
        print("="*60)
        
        plt.show(block=True)

def train_q_learning_agent(env, episodes=500):
    """Entraîne un agent Q-Learning"""
    print(f"\nEntraînement de l'agent Q-Learning ({episodes} épisodes)...")
    
    agent = QLearningAgent(env, alpha=0.3, gamma=0.95, epsilon=0.9)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Réduire epsilon progressivement
        epsilon = agent.update_epsilon(episode, episodes)
        
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        # Afficher la progression
        if (episode + 1) % 50 == 0:
            print(f"  Épisode {episode+1}: Récompense = {episode_reward:.2f}, ε = {epsilon:.3f}")
    
    print("✅ Entraînement terminé!")
    return agent

def demo_random_agent_interactive():
    """Démonstration interactive avec agent aléatoire"""
    print("\n" + "="*60)
    print("AGENT ALÉATOIRE INTERACTIF")
    print("="*60)
    
    env = GridWorldEnv(size=5)
    agent = RandomAgent(env)
    
    interactive = InteractiveGrid(env, agent, "Agent Aléatoire")
    interactive.interactive_mode()

def demo_value_iteration_interactive():
    """Démonstration interactive avec Value Iteration"""
    print("\n" + "="*60)
    print("VALUE ITERATION INTERACTIF")
    print("="*60)
    
    env = GridWorldEnv(size=5)
    
    print("Calcul de la solution optimale avec Value Iteration...")
    agent = ValueIterationAgent(env)
    
    print("✅ Solution optimale calculée!")
    
    interactive = InteractiveGrid(env, agent, "Value Iteration (Solution Optimale)")
    interactive.interactive_mode()

def demo_q_learning_interactive():
    """Démonstration interactive avec Q-Learning"""
    print("\n" + "="*60)
    print("Q-LEARNING INTERACTIF")
    print("="*60)
    
    env = GridWorldEnv(size=5)
    
    print("Options d'apprentissage:")
    print("1. Apprentissage rapide (500 épisodes)")
    print("2. Mode interactif sans apprentissage")
    
    choice = input("Votre choix (1-2): ").strip()
    
    if choice == "1":
        # Apprentissage rapide
        agent = train_q_learning_agent(env, episodes=500)
        
        # Réinitialiser pour l'interaction
        env.reset()
        interactive = InteractiveGrid(env, agent, "Q-Learning (Après apprentissage)")
        interactive.interactive_mode()
    
    elif choice == "2":
        # Mode interactif sans apprentissage
        agent = QLearningAgent(env, alpha=0.3, gamma=0.95, epsilon=0.9)
        interactive = InteractiveGrid(env, agent, "Q-Learning (Débutant)")
        interactive.interactive_mode()
    
    else:
        print("Choix invalide, utilisation de l'option 1 par défaut.")
        demo_q_learning_interactive()

def demo_moving_goal_interactive():
    """Démonstration avec but qui se déplace"""
    print("\n" + "="*60)
    print("BUT MOBILE INTERACTIF")
    print("="*60)
    
    env = GridWorldEnv(size=5)
    
    # Phase 1: Apprentissage avec premier but
    print("\nPhase 1: Apprentissage avec but en (4,4)")
    env.goal_pos = [4, 4]
    
    # Entraîner l'agent
    agent = train_q_learning_agent(env, episodes=300)
    
    # Test Phase 1
    print("\nTest Phase 1...")
    env.reset()
    interactive = InteractiveGrid(env, agent, "Phase 1: But en (4,4)")
    interactive.run_until_goal(max_steps=30, delay=0.2)
    plt.close()
    
    # Phase 2: But déplacé
    print("\n" + "="*40)
    print("Phase 2: But déplacé en (2,2)")
    print("="*40)
    
    env.move_goal([2, 2])
    env.reset()
    
    # Réentraîner l'agent
    print("\nRéentraînement pour le nouveau but...")
    agent = train_q_learning_agent(env, episodes=300)
    
    # Test Phase 2
    print("\nTest Phase 2...")
    env.reset()
    interactive = InteractiveGrid(env, agent, "Phase 2: But en (2,2)")
    interactive.run_until_goal(max_steps=30, delay=0.2)
    plt.close()

def auto_demo():
    """Démonstration automatique de tous les agents"""
    print("\n" + "="*60)
    print("DÉMONSTRATION AUTOMATIQUE COMPLÈTE")
    print("="*60)
    
    # Agent Aléatoire
    print("\n1. AGENT ALÉATOIRE")
    print("   L'agent se déplace au hasard...")
    env = GridWorldEnv(size=5)
    agent = RandomAgent(env)
    interactive = InteractiveGrid(env, agent, "Agent Aléatoire")
    interactive.run_until_goal(max_steps=30, delay=0.1)
    plt.close('all')
    time.sleep(1)
    
    # Value Iteration
    print("\n2. VALUE ITERATION")
    print("   Calcul de la solution optimale...")
    env = GridWorldEnv(size=5)
    agent = ValueIterationAgent(env)
    interactive = InteractiveGrid(env, agent, "Value Iteration")
    interactive.run_until_goal(max_steps=20, delay=0.2)
    plt.close('all')
    time.sleep(1)
    
    # Q-Learning
    print("\n3. Q-LEARNING")
    print("   Apprentissage en cours...")
    env = GridWorldEnv(size=5)
    agent = train_q_learning_agent(env, episodes=300)
    
    print("   Test de l'agent entraîné...")
    env.reset()
    interactive = InteractiveGrid(env, agent, "Q-Learning")
    interactive.run_until_goal(max_steps=20, delay=0.2)
    plt.close('all')
    
    print("\n" + "="*60)
    print("DÉMONSTRATION TERMINÉE!")
    print("="*60)

def interactive_menu():
    """Menu interactif principal"""
    while True:
        print("\n" + "="*60)
        print("GRIDWORLD INTERACTIF - MENU PRINCIPAL")
        print("="*60)
        print("1. Agent Aléatoire")
        print("2. Value Iteration (Solution optimale)")
        print("3. Q-Learning (Apprentissage par renforcement)")
        print("4. But Mobile (Adaptation en ligne)")
        print("5. Démonstration automatique complète")
        print("6. Quitter")
        print("="*60)
        
        choice = input("Votre choix (1-6): ").strip()
        
        if choice == "1":
            demo_random_agent_interactive()
        elif choice == "2":
            demo_value_iteration_interactive()
        elif choice == "3":
            demo_q_learning_interactive()
        elif choice == "4":
            demo_moving_goal_interactive()
        elif choice == "5":
            auto_demo()
        elif choice == "6":
            print("\nAu revoir!")
            break
        else:
            print("Choix invalide. Veuillez réessayer.")

def quick_start():
    """Démarrage rapide avec une démonstration simple"""
    print("="*60)
    print("GRIDWORLD - REINFORCEMENT LEARNING")
    print("="*60)
    
    print("\nBienvenue dans GridWorld!")
    print("\nDans ce monde, un agent (A) doit apprendre à atteindre un but (G)")
    print("en évitant les obstacles (X).")
    print("\nNous allons vous montrer comment différents agents résolvent ce problème.")
    
    print("\nOptions de démarrage:")
    print("1. Menu interactif complet")
    print("2. Démonstration rapide (recommandé)")
    print("3. Quitter")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
    if choice == "1":
        interactive_menu()
    elif choice == "2":
        auto_demo()
        # Après la démo, proposer le menu interactif
        continue_choice = input("\nVoulez-vous essayer le mode interactif? (o/n): ").strip().lower()
        if continue_choice == 'o' or continue_choice == 'oui':
            interactive_menu()
    elif choice == "3":
        print("\nAu revoir!")
    else:
        print("Choix invalide. Lancement de la démonstration rapide...")
        time.sleep(1)
        auto_demo()

if __name__ == "__main__":
    print("="*60)
    print("GRIDWORLD INTERACTIF - REINFORCEMENT LEARNING")
    print("="*60)
    
    print("\nChargement des modules...")
    
    # Vérifier que les modules nécessaires existent
    try:
        # Les imports sont déjà en haut du fichier
        print("✅ Modules chargés avec succès!")
    except ImportError as e:
        print(f"❌ Erreur de chargement: {e}")
        print("\nAssurez-vous d'avoir les fichiers suivants:")
        print("  - environment.py")
        print("  - agents.py")
        sys.exit(1)
    
    # Démarrer
    quick_start()