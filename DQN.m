classdef DQN < handle
    properties
        net
        target_net
        memory
        input_size
        num_actions
    end
    
    methods
        function obj = DQN()
            obj.input_size = Parameters.INPUT_SIZE;
            obj.num_actions = Parameters.NUM_ACTIONS;
            obj.net = obj.create_network();
            obj.target_net = obj.create_network();
            obj.memory = ReplayMemory(Parameters.MEMORY_CAPACITY);
            set(0, 'DefaultFigureVisible', 'off');
        end
        
        function net = create_network(obj)
            net = feedforwardnet([64, 64]);
            net = configure(net, zeros(obj.input_size, 1), zeros(obj.num_actions, 1));
            net.trainFcn = 'trainscg';
            net.performFcn = 'mse';
            net.trainParam.showWindow = false;
            net = init(net);
        end
        
        function action = select_action(obj, state, epsilon)
            if rand() < epsilon
                action = randi(obj.num_actions);
            else
                q_values = obj.net(state(:));
                [~, action] = max(q_values);
            end
        end
        
        function update(obj, batch_size)
            [states, actions, rewards, next_states, dones] = obj.memory.sample(batch_size);
            
            % 调试信息
            disp('States shape:');
            disp(size(states));
            disp('Actions shape:');
            disp(size(actions));
            disp('Rewards shape:');
            disp(size(rewards));
            disp('Next states shape:');
            disp(size(next_states));
            disp('Dones shape:');
            disp(size(dones));
            disp('Network input size:');
            disp(obj.net.inputs{1}.size);
            
            % 确保状态和下一个状态的格式正确
            states = reshape(states, obj.input_size, []);
            next_states = reshape(next_states, obj.input_size, []);
            
            % 计算目标Q值
            target_q_values = obj.target_net(next_states);
            [max_q_values, ~] = max(target_q_values, [], 1);
            target_q_values = rewards + Parameters.GAMMA * max_q_values .* (1 - dones);
            
            % 计算当前Q值
            current_q_values = obj.net(states);
            
            % 创建目标Q值矩阵
            target_f = current_q_values;
            for i = 1:batch_size
                target_f(actions(i), i) = target_q_values(i);
            end
            
            % 训练网络
            obj.net = train(obj.net, states, target_f, 'useGPU', 'no', 'showResources', 'no');
        end
        
        function update_target_network(obj)
            obj.target_net = obj.net;
        end
        
        function save_model(obj, filename)
            net = obj.net;
            save(filename, 'net');
        end
        
        function load_model(obj, filename)
            load(filename, 'net');
            obj.net = net;
            obj.target_net = net;
        end
    end
end