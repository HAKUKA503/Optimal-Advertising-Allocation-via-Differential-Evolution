clc;
clear all;
warning off;
tic;
format long;
format compact;
disp('DE/rand/1/bin for Advertising Allocation');

% -------------------- 问题参数设置 --------------------
n = 5;                  % 5个广告平台
popsize = 50;           % 种群规模
lower = 1;              % 单个平台最小投放金额（万元）
upper = 10;             % 单个平台最大投放金额（万元）
budget = 20;            % 总预算（万元）
lu = [lower * ones(1, n); upper * ones(1, n)];

% 平台转化模型参数 [a, b]
% a: 最大曝光潜力, b: 衰减系数
platformParams = [
    10000, 5e-1;  % 微博
    12000, 4e-1;  % B站
    8000, 6e-1;   % 小红书
    9000, 5e-1;   % 知乎
    15000, 3e-1;  % 抖音
];
a = platformParams(:, 1);
b = platformParams(:, 2);

% 罚函数系数
lambda1 = 10000;  % 总预算罚项
lambda2 = 10000;  % 范围罚项

% -------------------- 算法参数设置 --------------------
F = 0.7;              % 变异因子
CR = 0.9;             % 交叉率
maxGen = 500;         % 最大迭代次数

% 初始化记录变量
record_minval = zeros(maxGen, 1);          % 每代最大转化用户数
record_true_objval = zeros(maxGen, 1);     % 每代最优个体的真实目标函数值
best_solution = zeros(n, 1);
best_fitness = -inf;

% 初始化种群
popold = repmat(lu(1, :), popsize, 1) + rand(popsize, n) .* (repmat(lu(2, :) - lu(1, :), popsize, 1));

% 初始适应度评估
valParents = zeros(popsize, 1);
for i = 1:popsize
    valParents(i) = objectiveFunction(popold(i, :), a, b, lambda1, lambda2, budget);
end

% -------------------- 主循环 --------------------
for gen = 1:maxGen
    pop = popold;
    
    % 为每个个体生成r1, r2, r3
    r0 = 1:popsize;
    [r1, r2, r3] = gnR1R2R3(popsize, r0);
    
    % 变异操作
    vi = pop(r1, :) + F * (pop(r2, :) - pop(r3, :));
    vi = boundConstraint(vi, lu, budget);
    
    % 交叉操作
    mask = rand(popsize, n) > CR;
    jrand = randi(n, popsize, 1);
    for i = 1:popsize
        mask(i, jrand(i)) = false;  % 保证至少一个基因来自vi
    end
    ui = vi;
    ui(mask) = pop(mask);

    % 子代适应度
    valOffspring = zeros(popsize, 1);
    for i = 1:popsize
        valOffspring(i) = objectiveFunction(ui(i, :), a, b, lambda1, lambda2, budget);
    end
    
    % 选择操作
    improved = valOffspring < valParents;
    popold(improved, :) = ui(improved, :);
    valParents(improved) = valOffspring(improved);
    
    % 更新记录
    [current_min, min_idx] = min(valParents);
    record_minval(gen) = -current_min;  % 最大转化数

    % 记录真实目标函数值（已扣除罚项偏差，仅转化项）
    best_x = popold(min_idx, :);
    record_true_objval(gen) = -sum(a .* (1 - exp(-b .* best_x(:))));  % -f(x)

    if -current_min > best_fitness
        best_fitness = -current_min;
        best_solution = popold(min_idx, :)';
    end

    if mod(gen, 10) == 0
        fprintf('迭代 %d: 最大转化用户数 = %.2f\n', gen, best_fitness);
    end
end

% -------------------- 输出结果 --------------------
fprintf('\n最优广告投放策略:\n');
fprintf('平台\t投放金额(万元)\t转化用户数(人)\n');
total_cost = 0;
total_conversion = 0;
for i = 1:n
    conv = a(i) * (1 - exp(-b(i) * best_solution(i)));
    fprintf('%d\t%.2f\t\t%.2f\n', i, best_solution(i), conv);
    total_cost = total_cost + best_solution(i);
    total_conversion = total_conversion + conv;
end
fprintf('------------------------------------\n');
fprintf('总计\t%.2f\t\t%.2f\n', total_cost, total_conversion);
fprintf('预算利用率: %.2f%%\n', total_cost / budget * 100);

% -------------------- 收敛曲线 --------------------
figure;
plot(1:maxGen, record_true_objval, 'b-', 'LineWidth', 2);  % 蓝色实线
title('迭代次数-最优目标函数值曲线 (F=0.7, CR=0.9)');
xlabel('迭代次数');
ylabel('最优目标函数值 F_{best}^{(g)}');
grid on;

% -------------------- 函数定义 --------------------

% 目标函数
function f = objectiveFunction(x, a, b, lambda1, lambda2, budget)
    conversion = sum(a .* (1 - exp(-b .* x(:))));
    budget_penalty = lambda1 * max(0, sum(x) - budget)^2;
    range_penalty = lambda2 * sum((max(0, 1 - x).^2 + max(0, x - 10).^2));
    f = -conversion + budget_penalty + range_penalty;
end

% 边界与预算处理
function vi = boundConstraint(vi, lu, budget)
    xl = repmat(lu(1, :), size(vi, 1), 1);
    xu = repmat(lu(2, :), size(vi, 1), 1);
    vi = max(min(vi, 2 .* xu - vi), 2 .* xl - vi);
    vi = min(max(vi, xl), xu);
    for i = 1:size(vi, 1)
        sum_vi = sum(vi(i, :));
        if sum_vi > budget
            vi(i, :) = vi(i, :) * (budget / sum_vi);
        end
    end
end

% 三个随机索引生成函数
function [r1, r2, r3] = gnR1R2R3(popsize, r0)
    r1 = zeros(popsize, 1);
    r2 = zeros(popsize, 1);
    r3 = zeros(popsize, 1);
    for i = 1:popsize
        idx = r0(r0 ~= i);  % 排除自己
        perm = idx(randperm(length(idx)));
        r1(i) = perm(1);
        r2(i) = perm(2);
        r3(i) = perm(3);
    end
end

toc;