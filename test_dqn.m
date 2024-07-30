function [final_coverage, final_energy, final_delay, trajectory] = test_dqn(env, dqn, weights, num_episodes)
    total_coverage = 0;
    total_energy = 0;
    total_delay = 0;
    total_steps = 0;
    trajectory = [];

    for episode = 1:num_episodes
        state = env.reset();
        done = false;
        step = 0;

        while ~done && step < Parameters.MAX_STEPS
            action = dqn.select_action(state, 0);  % 使用 epsilon = 0 进行贪婪策略
            [next_state, ~, done] = env.step(action, weights);
            state = next_state;
            step = step + 1;

            total_coverage = total_coverage + env.coverage_rate;
            total_energy = total_energy + (1 - env.current_battery / env.max_battery);
            total_delay = total_delay + env.average_delay;
        end

        total_steps = total_steps + step;
        trajectory = [trajectory; env.trajectory];
    end

    final_coverage = total_coverage / total_steps;
    final_energy = total_energy / total_steps;
    final_delay = total_delay / total_steps;

    fprintf('Test Results:\n');
    fprintf('Average Coverage: %.4f\n', final_coverage);
    fprintf('Average Energy Consumption: %.4f\n', final_energy);
    fprintf('Average Delay: %.4f\n', final_delay);
    fprintf('Total Distance: %.2f m\n', env.total_distance);
    fprintf('Coverage Radius: %.2f m\n', env.coverage_radius);
end