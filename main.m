% 主程序
clc;
clear;
close all;

% 生成网格搜索权重
weights = Parameters.generate_grid_weights();

% 显示所有权重组合
fprintf('Generated weight combinations:\n');
for i = 1:size(weights, 1)
    fprintf('Weight %d: %.2f, %.2f, %.2f\n', i, weights(i,:));
end

% 显示总权重组合数
fprintf('\nTotal number of weight combinations: %d\n', size(weights, 1));

% 初始化总进度计数
total_combinations = length(Parameters.DIST_TYPES) * size(weights, 1);
combination_count = 0;

% 创建日志文件
log_file = fopen('training_log.txt', 'w');

% 对每种分布类型进行训练和测试
for dist_type_idx = 1:length(Parameters.DIST_TYPES)
    dist_type = Parameters.DIST_TYPES{dist_type_idx};
    fprintf('Training for distribution type: %s\n', dist_type);
    fprintf(log_file, 'Training for distribution type: %s\n', dist_type);
    
    % 创建环境
    env = UAVEnvironment(dist_type);
    
    % 创建结果数组
    results = cell(size(weights, 1), 1);
    
    % 串行训练
    for w = 1:size(weights, 1)
        weight_str = sprintf('%.2f, %.2f, %.2f', weights(w,:));
        fprintf('Starting training with weights: %s\n', weight_str);
        
        % 训练 DQN
        [dqn, stats] = train_dqn_wrapper(env, dqn, weights(w,:), dist_type);
        
        % 保存训练结果
        save(sprintf('results/dqn_%s_weight%d.mat', dist_type, w), 'dqn', 'stats');
        
        % 测试训练好的 DQN
        [final_coverage, final_energy, final_delay, trajectory] = test_dqn(env, dqn, weights(w,:), 100);
        
        % 保存测试结果
        save(sprintf('results/test_%s_weight%d.mat', dist_type, w), 'final_coverage', 'final_energy', 'final_delay', 'trajectory', 'weights');
        
        % 返回结果用于选择最优权重
        results{w} = struct('weights', weights(w,:), 'coverage', final_coverage, 'energy', final_energy, 'delay', final_delay);
        
        % 更新进度
        combination_count = combination_count + 1;
        fprintf('Progress: %d/%d combinations completed. Weights: %s, Coverage: %.4f, Energy: %.4f, Delay: %.4f\n', ...
                combination_count, total_combinations, weight_str, final_coverage, final_energy, final_delay);
        fprintf(log_file, 'Progress: %d/%d combinations completed. Weights: %s, Coverage: %.4f, Energy: %.4f, Delay: %.4f\n', ...
                combination_count, total_combinations, weight_str, final_coverage, final_energy, final_delay);
    end
    
    % 将 cell 数组转换为结构体数组
    results = [results{:}];
    
    % 选择最优权重
    [best_coverage, best_idx] = max([results.coverage]);
    best_weight = results(best_idx).weights;
    best_energy = results(best_idx).energy;
    best_delay = results(best_idx).delay;
    
    fprintf('Best weights for distribution type %s: %.2f, %.2f, %.2f\n', dist_type, best_weight);
    fprintf('Best performance: Coverage = %.4f, Energy = %.4f, Delay = %.4f\n', best_coverage, best_energy, best_delay);
    fprintf(log_file, 'Best weights for distribution type %s: %.2f, %.2f, %.2f\n', dist_type, best_weight);
    fprintf(log_file, 'Best performance: Coverage = %.4f, Energy = %.4f, Delay = %.4f\n', best_coverage, best_energy, best_delay);
    
    % 使用最优权重可视化结果
    visualize_results(env, dqn, best_weight);
end

fprintf('Training and testing completed.\n');
fprintf(log_file, 'Training and testing completed.\n');
fclose(log_file);