function visualize_results(env, dqn, weights)
    % 运行一个完整的回合
    state = env.reset();
    trajectory = [];
    coverages = [];
    energies = [];
    delays = [];
    
    for step = 1:Parameters.MAX_STEPS
        action = dqn.select_action(state, 0);  % 使用 epsilon = 0 进行贪婪策略
        [next_state, ~, done] = env.step(action, weights);
        state = next_state;
        
        trajectory = [trajectory; env.uav_pos];
        coverages = [coverages, env.coverage_rate];
        energies = [energies, 1 - env.current_battery / env.max_battery];
        delays = [delays, env.average_delay];
        
        if done
            break;
        end
    end
    
    % 创建图形
    figure('Position', [100, 100, 1200, 900]);
    
    % 绘制 UAV 轨迹和用户分布
    subplot(2, 3, 1);
    plot(env.users(:,1), env.users(:,2), 'b.', 'MarkerSize', 10);
    hold on;
    plot(trajectory(:,1), trajectory(:,2), 'r-', 'LineWidth', 2);
    plot(trajectory(1,1), trajectory(1,2), 'go', 'MarkerSize', 10, 'LineWidth', 2);  % 起始点
    plot(trajectory(end,1), trajectory(end,2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);  % 终止点
    title('UAV Trajectory and User Distribution');
    xlabel('X coordinate (m)');
    ylabel('Y coordinate (m)');
    legend('Users', 'UAV Trajectory', 'Start Point', 'End Point');
    axis([0 env.AREA_SIZE 0 env.AREA_SIZE]);
    grid on;
    
    % 绘制覆盖率变化
    subplot(2, 3, 2);
    plot(coverages, 'b-', 'LineWidth', 2);
    title('Coverage Rate over Time');
    xlabel('Time Step');
    ylabel('Coverage Rate');
    grid on;
    
    % 绘制能耗变化
    subplot(2, 3, 3);
    plot(energies, 'r-', 'LineWidth', 2);
    title('Energy Consumption over Time');
    xlabel('Time Step');
    ylabel('Energy Consumption (normalized)');
    grid on;
    
    % 绘制延迟变化
    subplot(2, 3, 4);
    plot(delays, 'g-', 'LineWidth', 2);
    title('Average Delay over Time');
    xlabel('Time Step');
    ylabel('Average Delay (s)');
    grid on;
    
    % 绘制用户密度热图
    subplot(2, 3, 5);
    [X, Y] = meshgrid(linspace(0, env.AREA_SIZE, 50), linspace(0, env.AREA_SIZE, 50));
    Z = zeros(size(X));
    for i = 1:length(env.users)
        Z = Z + exp(-((X-env.users(i,1)).^2 + (Y-env.users(i,2)).^2) / (2*(env.AREA_SIZE/10)^2));
    end
    contourf(X, Y, Z);
    colorbar;
    title('User Density Heatmap');
    xlabel('X coordinate');
    ylabel('Y coordinate');
    
    % 添加性能指标汇总
    subplot(2, 3, 6);
    text(0.1, 0.9, sprintf('Final Coverage: %.2f', coverages(end)), 'FontSize', 12);
    text(0.1, 0.7, sprintf('Final Energy Consumption: %.2f', energies(end)), 'FontSize', 12);
    text(0.1, 0.5, sprintf('Final Average Delay: %.2f', delays(end)), 'FontSize', 12);
    text(0.1, 0.3, sprintf('Total Distance: %.2f m', env.total_distance), 'FontSize', 12);
    text(0.1, 0.1, sprintf('Coverage Radius: %.2f m', env.coverage_radius), 'FontSize', 12);
    axis off;
    
    % 添加权重信息到标题
    sgtitle(sprintf('Results for Weights: %.2f, %.2f, %.2f', weights));
    
    % 调整子图之间的间距
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
    
    % 保存图形
    saveas(gcf, sprintf('results_%s_w%.2f_%.2f_%.2f.png', env.dist_type, weights(1), weights(2), weights(3)));
end