classdef ReplayMemory < handle
    properties
        capacity
        buffer
        position
        count
    end
    
    methods
        function obj = ReplayMemory(capacity)
            obj.capacity = capacity;
            obj.buffer = cell(1, capacity);
            obj.position = 0;
            obj.count = 0;
        end
        
        function add(obj, state, action, reward, next_state, done)
            obj.position = mod(obj.position, obj.capacity) + 1;
            obj.buffer{obj.position} = {state, action, reward, next_state, done};
            obj.count = min(obj.count + 1, obj.capacity);
        end
        
        function [states, actions, rewards, next_states, dones] = sample(obj, batch_size)
            indices = randi(obj.count, 1, batch_size);
            batch = obj.buffer(indices);
            
            % 预分配内存
            states = zeros(Parameters.INPUT_SIZE, batch_size);
            actions = zeros(1, batch_size);
            rewards = zeros(1, batch_size);
            next_states = zeros(Parameters.INPUT_SIZE, batch_size);
            dones = zeros(1, batch_size);
            
            % 填充数据
            for i = 1:batch_size
                states(:, i) = batch{i}{1};
                actions(i) = batch{i}{2};
                rewards(i) = batch{i}{3};
                next_states(:, i) = batch{i}{4};
                dones(i) = batch{i}{5};
            end
            
            % 确保所有输出都是双精度类型
            states = double(states);
            actions = double(actions);
            rewards = double(rewards);
            next_states = double(next_states);
            dones = double(dones);
        end
        
        function s = size(obj)
            s = obj.count;
        end
    end
end