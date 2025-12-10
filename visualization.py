import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import numpy as np

class RLVisualizer:
    """
    Visualisation pour Reinforcement Learning
    - Grille interactive
    - Q-values et valeurs d'état
    - Progression de l'apprentissage
    """
    
    @staticmethod
    def plot_environment(env, title="Environnement GridWorld"):
        """Affiche l'environnement de base"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Création de la grille
        grid = np.zeros((env.size, env.size))
        
        # Marquage des éléments
        grid[env.goal_pos[0], env.goal_pos[1]] = 2
        for obs in env.obstacles:
            grid[obs[0], obs[1]] = -1
        grid[env.agent_pos[0], env.agent_pos[1]] = 1
        
        # Colormap personnalisée
        cmap = colors.ListedColormap(['white', 'red', 'green', 'blue'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Affichage
        ax.imshow(grid, cmap=cmap, norm=norm, alpha=0.3)
        
        # Texte et annotations (SANS ÉMOJIS)
        for i in range(env.size):
            for j in range(env.size):
                if [i, j] == env.agent_pos:
                    ax.text(j, i, 'A', ha='center', va='center', 
                           fontsize=20, fontweight='bold', color='blue')
                elif [i, j] == env.goal_pos:
                    ax.text(j, i, 'G', ha='center', va='center', 
                           fontsize=20, fontweight='bold', color='green')
                elif [i, j] in env.obstacles:
                    ax.text(j, i, 'X', ha='center', va='center', 
                           fontsize=20, fontweight='bold', color='red')
                elif env.visited[i, j] > 0:
                    ax.text(j, i, str(min(9, int(env.visited[i, j]))), 
                           ha='center', va='center', fontsize=10, color='gray')
        
        # Configuration
        ax.set_xticks(np.arange(env.size))
        ax.set_yticks(np.arange(env.size))
        ax.set_xticklabels(np.arange(env.size))
        ax.set_yticklabels(np.arange(env.size))
        ax.grid(which='both', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Colonne', fontsize=12)
        ax.set_ylabel('Ligne', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_q_learning(env, agent, episode=None, step=None):
        """Affiche les Q-values et la politique"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Grille avec Q-values
        if agent.get_q_values() is not None:
            q_grid = agent.get_q_grid()
            im1 = axes[0].imshow(q_grid, cmap='viridis', alpha=0.7)
            plt.colorbar(im1, ax=axes[0], label='Max Q-value')
            
            # Valeurs numériques
            for i in range(env.size):
                for j in range(env.size):
                    if [i, j] not in env.obstacles:
                        axes[0].text(j, i, f'{q_grid[i, j]:.2f}', 
                                   ha='center', va='center', fontsize=8,
                                   color='white' if q_grid[i, j] < 5 else 'black')
        
        # Marquage des éléments (SANS ÉMOJIS)
        for i in range(env.size):
            for j in range(env.size):
                if [i, j] == env.agent_pos:
                    axes[0].text(j, i, 'A', ha='center', va='center', 
                               fontsize=20, fontweight='bold', color='blue')
                elif [i, j] == env.goal_pos:
                    axes[0].text(j, i, 'G', ha='center', va='center', 
                               fontsize=20, fontweight='bold', color='green')
                elif [i, j] in env.obstacles:
                    axes[0].add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                                      fill=True, color='red', alpha=0.3))
                    axes[0].text(j, i, 'X', ha='center', va='center', 
                               fontsize=15, fontweight='bold', color='darkred')
        
        axes[0].set_title(f'Q-values (Episode {episode}, Step {step})', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Colonne')
        axes[0].set_ylabel('Ligne')
        
        # 2. Politique
        axes[1].imshow(np.zeros((env.size, env.size)), cmap='gray', alpha=0.1)
        
        for i in range(env.size):
            for j in range(env.size):
                state = i * env.size + j
                
                if [i, j] == env.goal_pos:
                    axes[1].text(j, i, 'G', ha='center', va='center', 
                               fontsize=20, fontweight='bold', color='green')
                elif [i, j] in env.obstacles:
                    axes[1].text(j, i, 'X', ha='center', va='center', 
                               fontsize=15, fontweight='bold', color='red')
                else:
                    # Afficher la meilleure action
                    if agent.get_q_values() is not None:
                        q_values = agent.get_q_values()[state]
                        if not np.all(q_values == 0):
                            best_action = np.argmax(q_values)
                            axes[1].text(j, i, env.action_names[best_action], 
                                       ha='center', va='center', fontsize=24,
                                       fontweight='bold', color='blue')
        
        axes[1].set_title('Politique Apprise', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Colonne')
        axes[1].set_ylabel('Ligne')
        
        # 3. Visites
        visits_grid = env.visited.copy()
        # Normaliser pour une meilleure visualisation
        if np.max(visits_grid) > 0:
            visits_grid = visits_grid / np.max(visits_grid)
        
        im3 = axes[2].imshow(visits_grid, cmap='Reds', alpha=0.7)
        plt.colorbar(im3, ax=axes[2], label='Visites (normalisées)')
        
        for i in range(env.size):
            for j in range(env.size):
                if env.visited[i, j] > 0:
                    axes[2].text(j, i, str(int(env.visited[i, j])), 
                               ha='center', va='center', fontsize=10,
                               color='black' if visits_grid[i, j] < 0.5 else 'white')
        
        axes[2].set_title('Exploration de l\'espace', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Colonne')
        axes[2].set_ylabel('Ligne')
        
        plt.suptitle(f'Agent: {agent.name} | ε={agent.epsilon:.3f}' 
                     if hasattr(agent, 'epsilon') else f'Agent: {agent.name}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_learning_progress(rewards_history, steps_history, epsilon_history=None):
        """Affiche la progression de l'apprentissage"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Récompenses cumulatives
        axes[0, 0].plot(rewards_history, alpha=0.6, color='green', linewidth=1)
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Récompense cumulée')
        axes[0, 0].set_title('Évolution des Récompenses', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Moyenne glissante
        window = max(1, len(rewards_history) // 20)
        if len(rewards_history) >= window:
            moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(rewards_history)), moving_avg, 
                          linewidth=2, color='red', label=f'Moyenne ({window} épisodes)')
            axes[0, 0].legend()
        
        # 2. Nombre de pas par épisode
        axes[0, 1].plot(steps_history, alpha=0.6, color='blue', linewidth=1)
        axes[0, 1].set_xlabel('Épisode')
        axes[0, 1].set_ylabel('Pas nécessaires')
        axes[0, 1].set_title('Performance par Épisode', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Moyenne glissante pour les pas
        if len(steps_history) >= window:
            moving_avg_steps = np.convolve(steps_history, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(steps_history)), moving_avg_steps, 
                          linewidth=2, color='orange', label=f'Moyenne ({window} épisodes)')
            axes[0, 1].legend()
        
        # 3. Distribution des récompenses
        if len(rewards_history) > 1:
            axes[1, 0].hist(rewards_history, bins=min(20, len(rewards_history)), 
                           alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(np.mean(rewards_history), color='red', linestyle='--', 
                             label=f'Moyenne: {np.mean(rewards_history):.2f}')
            axes[1, 0].axvline(np.median(rewards_history), color='blue', linestyle='--',
                             label=f'Médiane: {np.median(rewards_history):.2f}')
            axes[1, 0].set_xlabel('Récompense')
            axes[1, 0].set_ylabel('Fréquence')
            axes[1, 0].set_title('Distribution des Récompenses', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Pas assez de données\npour l\'histogramme',
                          ha='center', va='center', transform=axes[1, 0].transAxes,
                          fontsize=12)
            axes[1, 0].set_title('Distribution des Récompenses', fontsize=12, fontweight='bold')
        
        # 4. Epsilon (si Q-Learning)
        if epsilon_history and len(epsilon_history) > 1:
            axes[1, 1].plot(epsilon_history, color='purple', linewidth=2)
            axes[1, 1].set_xlabel('Épisode')
            axes[1, 1].set_ylabel('ε (taux d\'exploration)')
            axes[1, 1].set_title('Évolution d\'Epsilon (Exploration)', 
                                fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Non applicable\n(Agent sans exploration)',
                          ha='center', va='center', transform=axes[1, 1].transAxes,
                          fontsize=12)
            axes[1, 1].set_title('Évolution d\'Epsilon', fontsize=12, fontweight='bold')
        
        plt.suptitle('Analyse de l\'Apprentissage par Renforcement', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_agent_comparison(results):
        """Compare les performances de différents agents"""
        agents = list(results.keys())
        avg_rewards = [np.mean(results[agent]['rewards']) for agent in agents]
        avg_steps = [np.mean(results[agent]['steps']) for agent in agents]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Comparaison des récompenses
        colors = ['red', 'green', 'blue', 'orange']
        bars1 = axes[0].bar(range(len(agents)), avg_rewards, 
                           color=colors[:len(agents)], alpha=0.7)
        axes[0].set_xticks(range(len(agents)))
        axes[0].set_xticklabels(agents, rotation=45, ha='right')
        axes[0].set_ylabel('Récompense moyenne')
        axes[0].set_title('Comparaison des Récompenses', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # Comparaison des pas
        bars2 = axes[1].bar(range(len(agents)), avg_steps, 
                           color=colors[:len(agents)], alpha=0.7)
        axes[1].set_xticks(range(len(agents)))
        axes[1].set_xticklabels(agents, rotation=45, ha='right')
        axes[1].set_ylabel('Pas moyens par épisode')
        axes[1].set_title('Comparaison de l\'Efficacité', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom')
        
        plt.suptitle('Comparaison des Agents', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()