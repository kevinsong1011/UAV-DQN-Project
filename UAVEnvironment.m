classdef UAVEnvironment < handle
    properties (Constant)
        AREA_SIZE = Parameters.AREA_SIZE;
        NUM_USERS = Parameters.NUM_USERS;
        STEP_SIZE = Parameters.STEP_SIZE;
    end
    
    properties
        users
        uav_pos
        dist_type
        
        % 系统模型参数
        Pt = 0.1;    % 发射功率 (W)
        PL0 = 40;    % 参考距离处的路径损耗 (dB)
        gamma = 3;   % 路径损耗指数
        d0 = 1;      % 参考距离 (m)
        Pr_threshold = -80;  % 接收信号功率阈值 (dBm)
        
        % 能耗和电池参数
        max_battery = 1000;  % 最大电池容量 (J)
        current_battery     % 当前电池电量 (J)
        kfly = 0.2;  % 飞行能耗系数 (J/m)
        kcomm = 1e-6;  % 通信能耗系数 (J/bit)
        delta_t = 1;  % 决策间隔 (s)
        
        % 延时模型参数
        B = 1e6;      % 带宽 (Hz)
        Pn = -104;    % 噪声功率 (dBm)
        max_delay = 1;  % 最大延时 (s)
        delay_factor = 0.007;  % 延时因子 (s/m)
        
        % 轨迹和覆盖率
        trajectory = [];  % 用于记录 UAV 轨迹
        previous_coverage = 0;  % 用于记录上一步的覆盖率
        
        % 性能指标属性
        coverage_rate
        average_delay
        coverage_radius  % 覆盖半径
        total_distance
    end
    
    methods
        function obj = UAVEnvironment(dist_type)
            obj.dist_type = dist_type;
            obj.reset();
        end

        function delay = calculate_delay(obj)
            distances = pdist2(obj.uav_pos, obj.users);
            Pr = obj.calculate_received_power(distances);
            covered_users = Pr >= obj.Pr_threshold;
            
            if sum(covered_users) == 0
                delay = obj.max_delay;
            else
                covered_distances = distances(covered_users);
                delays = min(obj.delay_factor * covered_distances, obj.max_delay);
                delay = mean(delays);
            end
        end

        function Pr = calculate_received_power(obj, distances)
            path_loss = obj.PL0 + 10 * obj.gamma * log10(distances / obj.d0);
            Pr = 10*log10(obj.Pt*1000) - path_loss;  % Convert W to dBm
        end

        function coverage = calculate_coverage(obj)
            distances = pdist2(obj.uav_pos, obj.users);
            Pr = obj.calculate_received_power(distances);
            covered_users = sum(Pr >= obj.Pr_threshold);
            coverage = covered_users / obj.NUM_USERS;

            % 计算覆盖半径
            obj.coverage_radius = (10^((10*log10(obj.Pt*1000) - obj.PL0 - obj.Pr_threshold) / (10 * obj.gamma))) * obj.d0;
        end

        function [next_state, reward, done] = step(obj, action, weights)
            prev_pos = obj.uav_pos;
            obj.uav_pos = obj.move_uav(action);
            
            distance_moved = norm(obj.uav_pos - prev_pos);
            obj.total_distance = obj.total_distance + distance_moved;
            
            obj.coverage_rate = obj.calculate_coverage();
            energy_consumed = obj.calculate_energy(distance_moved);
            obj.current_battery = max(0, obj.current_battery - energy_consumed);
            
            obj.average_delay = obj.calculate_delay();
            
            % 归一化处理
            normalized_coverage = obj.coverage_rate;
            normalized_energy = 1 - (obj.current_battery / obj.max_battery);  % 能耗用电池消耗量表示
            normalized_delay = min(obj.average_delay / obj.max_delay, 1);
            
            % 奖励计算
            coverage_reward = weights(1) * normalized_coverage;
            energy_penalty = weights(2) * normalized_energy;
            delay_penalty = weights(3) * normalized_delay;

            reward = coverage_reward - energy_penalty - delay_penalty;

            % 添加额外的奖励以鼓励探索
            if obj.coverage_rate > obj.previous_coverage
                reward = reward + 0.1;
            end
            obj.previous_coverage = obj.coverage_rate;
            
            if obj.current_battery <= 0
                reward = reward - 10;  % Penalty for depleting battery
                done = true;
            else
                done = obj.total_distance >= obj.AREA_SIZE * 2;
            end
            
            obj.trajectory = [obj.trajectory; obj.uav_pos];
            
            next_state = obj.get_state();
        end

        function state = get_state(obj)
    % 1-2. UAV相对位置（相对于区域中心的归一化坐标）
    relative_pos = (obj.uav_pos - obj.AREA_SIZE/2) / (obj.AREA_SIZE/2);
    
    % 3. 覆盖率
    coverage = obj.coverage_rate;
    
    % 4. 归一化平均延迟
    normalized_delay = min(obj.average_delay / obj.max_delay, 1);
    
    % 5. 归一化电池电量（同时反映了能耗情况）
    battery_level = obj.current_battery / obj.max_battery;
    
    % 6-9. 用户分布信息（四个方向的用户密度）
    densities = obj.calculate_quadrant_densities();
    
    % 确保所有元素都是列向量
    state = [relative_pos(:); coverage; normalized_delay; battery_level; densities(:)];
end

        function densities = calculate_quadrant_densities(obj)
            center = obj.AREA_SIZE / 2;
            quadrants = zeros(4, 1);
            for i = 1:obj.NUM_USERS
                if obj.users(i,1) <= center
                    if obj.users(i,2) <= center
                        quadrants(1) = quadrants(1) + 1;
                    else
                        quadrants(2) = quadrants(2) + 1;
                    end
                else
                    if obj.users(i,2) <= center
                        quadrants(3) = quadrants(3) + 1;
                    else
                        quadrants(4) = quadrants(4) + 1;
                    end
                end
            end
            densities = quadrants / obj.NUM_USERS;
        end

        function state = reset(obj)
            obj.users = obj.generate_users();
            obj.uav_pos = [0, 0];  % 初始位置设置为 (0, 0)
            obj.total_distance = 0;
            obj.current_battery = obj.max_battery;
            obj.trajectory = obj.uav_pos;
            obj.previous_coverage = 0;
            obj.coverage_rate = 0;
            obj.average_delay = obj.max_delay;
            obj.coverage_radius = 0;
            state = obj.get_state();
        end

        function next_uav_pos = move_uav(obj, action)
            directions = [0, 1; 1, 1; 1, 0; 1, -1; 0, -1; -1, -1; -1, 0; -1, 1; 0, 0];
            next_uav_pos = obj.uav_pos + obj.STEP_SIZE * directions(action, :);
            next_uav_pos = min(max(next_uav_pos, 0), obj.AREA_SIZE);
        end

        function energy = calculate_energy(obj, distance_moved)
            % 计算飞行能耗
            Efly = obj.kfly * distance_moved;
            
            % 计算通信能耗
            covered_users = obj.calculate_coverage() * obj.NUM_USERS;
            total_bits = obj.B * obj.delta_t * covered_users;
            Ecomm = obj.kcomm * total_bits;
            
            % 总能耗
            energy = Efly + Ecomm;
        end

        function users = generate_users(obj)
            switch obj.dist_type
                case 'Uniform Random Distribution'
                    users = obj.AREA_SIZE * rand(obj.NUM_USERS, 2);
                case 'Random Linear Distribution'
                    x = linspace(0, obj.AREA_SIZE, obj.NUM_USERS)';
                    y = obj.AREA_SIZE * rand(obj.NUM_USERS, 1);
                    users = [x, y];
                case 'Center-Clustered Distribution'
                    center = obj.AREA_SIZE / 2;
                    std_dev = obj.AREA_SIZE / 6;
                    users = center + std_dev * randn(obj.NUM_USERS, 2);
                    users = min(max(users, 0), obj.AREA_SIZE);
                case 'Multi-Cluster Distribution'
                    num_clusters = 5;
                    cluster_centers = obj.AREA_SIZE * rand(num_clusters, 2);
                    cluster_std_dev = obj.AREA_SIZE / 20;
                    users = [];
                    for i = 1:num_clusters
                        cluster_users = cluster_centers(i, :) + cluster_std_dev * randn(floor(obj.NUM_USERS / num_clusters), 2);
                        users = [users; cluster_users];
                    end
                    users = min(max(users, 0), obj.AREA_SIZE);
                otherwise
                    error('Unknown distribution type');
            end
        end
    end
end