classdef Parameters
    properties (Constant)
        % 环境参数
        AREA_SIZE = 1000  % m
        NUM_USERS = 100
        
        % UAV 参数
        STEP_SIZE = 50  % m
        
        % DQN 参数
        INPUT_SIZE = 9  % 新的状态向量大小
        NUM_ACTIONS = 9
        MEMORY_CAPACITY = 10000
        BATCH_SIZE = 32
        GAMMA = 0.99
        EPSILON_START = 1.0
        EPSILON_END = 0.01
        EPSILON_DECAY = 0.995
        LEARNING_RATE = 0.001
        
        % 训练参数
        NUM_EPISODES = 1000
        MAX_STEPS = 200
        
        % 分布类型
        DIST_TYPES = {'Uniform Random Distribution', 'Random Linear Distribution', 'Center-Clustered Distribution', 'Multi-Cluster Distribution'}
        
        % 网格搜索权重范围
        WEIGHT_RANGE = 0.1:0.2:1
    end
    
    methods (Static)
        function weights = generate_grid_weights()
            [w1, w2, w3] = ndgrid(Parameters.WEIGHT_RANGE, Parameters.WEIGHT_RANGE, Parameters.WEIGHT_RANGE);
            weights = [w1(:), w2(:), w3(:)];
            weights = weights(abs(sum(weights, 2) - 1) < 1e-6, :);  % 确保权重和为1
            
            % 添加初始权重 1:1:1
            initial_weight = [1/3, 1/3, 1/3];
            weights = [initial_weight; weights];
            
            % 去除重复的权重组合
            weights = unique(weights, 'rows');
        end
    end
end