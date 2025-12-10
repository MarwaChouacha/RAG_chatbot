import numpy as np
import time
from environment import GridWorldEnv
from agents import RandomAgent, ValueIterationAgent, QLearningAgent
from visualization import RLVisualizer

def run_random_agent_demo():
    """Démonstration de l'agent aléatoire"""
    print("\n" + "="*70)
    print("PHASE 1: AGENT ALÉATOIRE - Pas d'apprentissage")
    print("="*70)
    
    # Création de l'environnement
    env = GridWorldEnv(size=5)
    
    # Affichage initial
    RLVisualizer.plot_environment(env, "Environnement Initial")
    print("\nConfiguration initiale:")
    print(env.render())
    
    # Création de l'agent
    agent = RandomAgent(env)
    
    # Exécution de quelques épisodes
    episodes = 3
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50
        done = False
        
        print(f"\n🎮 Épisode {episode + 1}:")
        
        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Affichage du progrès
            if steps % 10 == 0:
                print(f"  Step {steps}: Position {env.agent_pos}, Récompense: {reward:.2f}")
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Visualisation des premiers pas
            if steps <= 3:
                RLVisualizer.plot_q_learning(env, agent, episode+1, steps)
                time.sleep(0.5)
        
        # Résultats de l'épisode
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        print(f"  🏁 Terminé en {steps} pas, Récompense totale: {total_reward:.2f}")
        if done:
            print("  ✓ Objectif atteint!" if reward > 0 else "  ✗ Échec!")
    
    return {
        'rewards': rewards_history,
        'steps': steps_history,
        'agent': agent
    }

def run_value_iteration_demo():
    """Démonstration de Value Iteration"""
    print("\n" + "="*70)
    print("PHASE 2: VALUE ITERATION - Planification optimale")
    print("="*70)
    
    # Création de l'environnement
    env = GridWorldEnv(size=5)
    
    print("\n🎯 Apprentissage en cours (Value Iteration)...")
    
    # Création de l'agent (apprentissage immédiat)
    agent = ValueIterationAgent(env)
    
    print("✅ Apprentissage terminé!")
    print(f"\n📊 Valeurs d'état optimales:")
    print(env.state_values)
    
    # Visualisation des valeurs
    RLVisualizer.plot_environment(env, "Valeurs d'État Optimales")
    
    # Test de la politique optimale
    episodes = 2
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 20
        done = False
        
        print(f"\n🎮 Épisode {episode + 1} (Politique optimale):")
        
        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Afficher l'action
            x, y = env.get_coords(state)
            print(f"  Step {steps}: Position ({x},{y}) -> {env.action_names[action]}")
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Visualisation
            if steps <= 5:
                RLVisualizer.plot_q_learning(env, agent, episode+1, steps)
                time.sleep(0.5)
        
        # Résultats
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        print(f"  🏁 Terminé en {steps} pas, Récompense: {total_reward:.2f}")
    
    return {
        'rewards': rewards_history,
        'steps': steps_history,
        'agent': agent
    }

def run_q_learning_demo(fixed_goal=True, moving_goal=False):
    """Démonstration de Q-Learning avec/sans but mobile"""
    if fixed_goal:
        print("\n" + "="*70)
        print("PHASE 3: Q-LEARNING - Apprentissage par renforcement")
        print("="*70)
        goal_type = "fixe"
    else:
        print("\n" + "="*70)
        print("PHASE 4: Q-LEARNING AVEC BUT MOBILE - Adaptation en ligne")
        print("="*70)
        goal_type = "mobile"
    
    # Création de l'environnement
    env = GridWorldEnv(size=5)
    
    # Création de l'agent
    agent = QLearningAgent(env, alpha=0.2, gamma=0.9, epsilon=0.8)
    
    # Phase d'apprentissage
    learning_episodes = 200 if fixed_goal else 300
    print(f"\n🧠 Phase d'apprentissage ({learning_episodes} épisodes, but {goal_type})...")
    
    rewards_history = []
    steps_history = []
    epsilon_history = []
    
    for episode in range(learning_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        done = False
        
        # Déplacer le but au milieu de l'apprentissage (si mobile)
        if moving_goal and episode == learning_episodes // 2:
            env.move_goal([2, 2])  # Nouvelle position au centre
            agent.reset_learning()  # Réinitialise partiellement l'apprentissage
        
        # Exploration vs exploitation
        current_epsilon = agent.update_epsilon(episode, learning_episodes)
        epsilon_history.append(current_epsilon)
        
        while not done and steps < max_steps:
            # Choix d'action
            action = agent.choose_action(state, current_epsilon)
            
            # Exécution
            next_state, reward, done = env.step(action)
            
            # Apprentissage
            agent.learn(state, action, reward, next_state, done)
            
            # Mise à jour
            state = next_state
            total_reward += reward
            steps += 1
        
        # Enregistrement des statistiques
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        # Affichage périodique
        if (episode + 1) % 50 == 0:
            print(f"  Épisode {episode+1}: Récompense={total_reward:.2f}, "
                  f"Pas={steps}, ε={current_epsilon:.3f}")
    
    print("✅ Apprentissage terminé!")
    
    # Visualisation de la progression
    RLVisualizer.plot_learning_progress(rewards_history, steps_history, epsilon_history)
    
    # Phase de test
    print("\n🧪 Phase de test (exploitation pure, ε=0)...")
    test_episodes = 5
    test_rewards = []
    test_steps = []
    
    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50
        done = False
        
        print(f"\n🎮 Test {episode + 1}:")
        
        while not done and steps < max_steps:
            action = agent.choose_action(state, epsilon=0)  # Pas d'exploration
            next_state, reward, done = env.step(action)
            
            if steps % 5 == 0:
                x, y = env.get_coords(state)
                print(f"  Step {steps}: Position ({x},{y}) -> {env.action_names[action]}")
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Visualisation
            if steps <= 10:
                RLVisualizer.plot_q_learning(env, agent, episode+1, steps)
                time.sleep(0.3)
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        
        print(f"  🏁 Terminé en {steps} pas, Récompense: {total_reward:.2f}")
        if moving_goal:
            print(f"  🎯 But actuel: {env.goal_pos}")
    
    # Affichage final
    print(f"\n📊 Résultats du test (but {goal_type}):")
    print(f"  Récompense moyenne: {np.mean(test_rewards):.2f}")
    print(f"  Pas moyens: {np.mean(test_steps):.1f}")
    
    if moving_goal:
        print(f"\n💡 Observation: L'agent a dû s'adapter au déplacement du but!")
    
    return {
        'rewards': test_rewards,
        'steps': test_steps,
        'agent': agent,
        'learning_history': rewards_history
    }

def adaptive_challenge():
    """Défi d'adaptation: l'agent doit trouver un but qui se déplace aléatoirement"""
    print("\n" + "="*70)
    print("DÉFI: AGENT ADAPTATIF - But se déplaçant aléatoirement")
    print("="*70)
    
    env = GridWorldEnv(size=5)
    agent = QLearningAgent(env, alpha=0.3, gamma=0.95, epsilon=0.7)
    
    # Positions possibles pour le but (éviter les obstacles)
    possible_goals = []
    for i in range(env.size):
        for j in range(env.size):
            if [i, j] not in env.obstacles and [i, j] != [0, 0]:
                possible_goals.append([i, j])
    
    episodes = 400
    print(f"\n🧠 Apprentissage adaptatif ({episodes} épisodes)...")
    
    for episode in range(episodes):
        # Changer le but périodiquement
        if episode % 50 == 0 and episode > 0:
            new_goal = possible_goals[np.random.randint(len(possible_goals))]
            env.move_goal(new_goal)
            print(f"  Épisode {episode}: Nouveau but à {new_goal}")
        
        # Épisode d'apprentissage
        state = env.reset()
        done = False
        steps = 0
        max_steps = 100
        
        current_epsilon = agent.update_epsilon(episode, episodes)
        
        while not done and steps < max_steps:
            action = agent.choose_action(state, current_epsilon)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        
        if (episode + 1) % 100 == 0:
            print(f"  Épisode {episode+1}: ε={current_epsilon:.3f}")
    
    # Test final
    print("\n🧪 Test final de l'agent adaptatif...")
    test_goals = [[4, 4], [2, 2], [0, 4], [4, 0]]
    
    for i, goal in enumerate(test_goals):
        env.move_goal(goal)
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        print(f"\n🎯 Test avec but à {goal}:")
        
        while not done and steps < 30:
            action = agent.choose_action(state, epsilon=0)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
        
        print(f"  {'✓' if done else '✗'} {'Trouvé' if done else 'Non trouvé'} en {steps} pas")
        print(f"  Récompense: {total_reward:.2f}")
        
        # Visualisation
        RLVisualizer.plot_q_learning(env, agent, i+1, steps)
        time.sleep(0.5)
    
    print("\n🏆 Défi terminé! L'agent a démontré sa capacité d'adaptation.")

def main():
    """Fonction principale"""
    print("="*70)
    print("SYSTÈME DE REINFORCEMENT LEARNING - GridWorld")
    print("="*70)
    
    # Résultats de tous les agents
    results = {}
    
    # Phase 1: Agent aléatoire
    random_results = run_random_agent_demo()
    results['Aléatoire'] = random_results
    
    # Phase 2: Value Iteration
    vi_results = run_value_iteration_demo()
    results['Value Iteration'] = vi_results
    
    # Phase 3: Q-Learning avec but fixe
    ql_fixed_results = run_q_learning_demo(fixed_goal=True, moving_goal=False)
    results['Q-Learning (fixe)'] = ql_fixed_results
    
    # Phase 4: Q-Learning avec but mobile
    ql_mobile_results = run_q_learning_demo(fixed_goal=False, moving_goal=True)
    results['Q-Learning (mobile)'] = ql_mobile_results
    
    # Défi adaptatif
    adaptive_challenge()
    
    # Comparaison finale
    print("\n" + "="*70)
    print("ANALYSE COMPARATIVE DES AGENTS")
    print("="*70)
    
    comparison_data = {}
    for name, data in results.items():
        if 'rewards' in data:
            comparison_data[name] = {
                'rewards': data['rewards'],
                'steps': data['steps']
            }
            print(f"\n{name}:")
            print(f"  Récompense moyenne: {np.mean(data['rewards']):.2f}")
            print(f"  Pas moyens: {np.mean(data['steps']):.1f}")
    
    # Visualisation comparative
    RLVisualizer.plot_agent_comparison(comparison_data)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\n1. Agent Aléatoire:")
    print("   - Pas d'apprentissage")
    print("   - Performances imprévisibles")
    print("   - Ne trouve pas systématiquement le but")
    
    print("\n2. Value Iteration:")
    print("   - Apprentissage optimal hors-ligne")
    print("   - Nécessite un modèle de l'environnement")
    print("   - Ne s'adapte pas aux changements")
    
    print("\n3. Q-Learning:")
    print("   - Apprentissage en ligne par essai-erreur")
    print("   - Pas besoin de modèle de l'environnement")
    print("   - S'adapte au déplacement du but")
    print("   - Exploration vs exploitation (epsilon-greedy)")
    
    print("\n✅ L'agent Q-Learning démontre une véritable capacité d'apprentissage")
    print("   et d'adaptation, caractéristique du Reinforcement Learning!")

if __name__ == "__main__":
    main()