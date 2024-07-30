function [dqn, stats] = train_dqn_wrapper(env, dqn, weights, dist_type)
    % Initialize replay memory
    memory = ReplayMemory(Parameters.MEMORY_CAPACITY);

    % Initialize statistics
    stats = struct;
    stats.episode_rewards = zeros(1, Parameters.NUM_EPISODES);
    stats.episode_coverages = zeros(1, Parameters.NUM_EPISODES);
    stats.episode_energies = zeros(1, Parameters.NUM_EPISODES);
    stats.episode_delays = zeros(1, Parameters.NUM_EPISODES);
    stats.q_values = zeros(1, Parameters.NUM_EPISODES);
    stats.losses = zeros(1, Parameters.NUM_EPISODES);

    for episode = 1:Parameters.NUM_EPISODES
        % Reset environment and get initial state
        state = env.reset();

        for step = 1:Parameters.MAX_STEPS
            % Select action using epsilon-greedy policy
            if rand() < Parameters.EPSILON
                action = randi(length(Parameters.ACTIONS));
            else
                action = dqn.select_action(state);
            end

            % Take action and observe next state and reward
            [next_state, reward, done] = env.step(action, weights);

            % Store experience in replay memory
            memory.add(state, action, reward, next_state, done);

            % Update statistics
            stats.episode_rewards(episode) = stats.episode_rewards(episode) + reward;
            stats.episode_coverages(episode) = env.coverage_ratio;
            stats.episode_energies(episode) = env.energy_consumption;
            stats.episode_delays(episode) = env.avg_delay;

            % Train DQN if memory has enough samples
            if memory.count > Parameters.BATCH_SIZE
                experiences = memory.sample(Parameters.BATCH_SIZE);
                [q_values, loss] = dqn.train(experiences);
                stats.q_values(episode) = mean(q_values);
                stats.losses(episode) = loss;
            end

            % Update state
            state = next_state;

            % Break loop if episode is done
            if done
                break;
            end
        end

        if mod(episode, 10) == 0
            fprintf('User Distribution: %s | Weights: %.2f, %.2f, %.2f\n', dist_type, weights(1), weights(2), weights(3));
            fprintf('Episode: %d | Steps: %d | Avg. Coverage: %.4f | Avg. Energy: %.4f | Avg. Delay: %.4f | Avg. Reward: %.4f\n', ...
                episode, step, mean(stats.episode_coverages(episode-9:episode)), ...
                mean(stats.episode_energies(episode-9:episode)), mean(stats.episode_delays(episode-9:episode)), ...
                mean(stats.episode_rewards(episode-9:episode)));
            fprintf('Avg. Q-value: %.4f | Loss: %.4f\n', mean(stats.q_values(episode-9:episode)), mean(stats.losses(episode-9:episode)));
            fprintf('------------------------\n');
        end
    end
    
    % Return DQN and statistics
    stats.avg_rewards = mean(stats.episode_rewards);
    stats.avg_coverages = mean(stats.episode_coverages);
    stats.avg_energies = mean(stats.episode_energies);
    stats.avg_delays = mean(stats.episode_delays);
end