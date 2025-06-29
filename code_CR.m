clc;
clear;
warning off;
tic;
format long;
format compact;

% ========== 基础问题设置 ==========
n = 5;
popsize = 50;
lower = 1;
upper = 10;
budget = 20;
lu = [lower * ones(1, n); upper * ones(1, n)];

platformParams = [
    10000, 5e-1;  % 微博
    12000, 4e-1;  % B站
    8000, 6e-1;   % 小红书
    9000, 5e-1;   % 知乎
    15000, 3e-1;  % 抖音
];
a = platformParams(:, 1);
b = platformParams(:, 2);

lambda1 = 10000;
lambda2 = 10000;

F = 0.7;             % 固定变异因子
maxGen = 500;
CR_list = [0.1, 0.3, 0.5, 0.7, 0.9];
run_times = 10;

mean_curves = zeros(length(CR_list), maxGen);
final_vals = zeros(length(CR_list), run_times);
converge_iters = zeros(length(CR_list), run_times);

fprintf('开始CR敏感性对比实验...\n');

for cr_idx = 1:length(CR_list)
    CR = CR_list(cr_idx);
    fprintf('CR = %.1f\n', CR);
    curves = zeros(run_times, maxGen);

    for run = 1:run_times
        % 初始化
        pop = repmat(lu(1, :), popsize, 1) + rand(popsize, n) .* (repmat(lu(2, :) - lu(1, :), popsize, 1));
        val = zeros(popsize, 1);
        for i = 1:popsize
            val(i) = obj(pop(i, :));
        end

        best_curve = zeros(maxGen, 1);
        diversity50 = 0;

        for gen = 1:maxGen
            r0 = 1:popsize;
            [r1, r2, r3] = rand3(popsize);
            vi = pop(r1,:) + F*(pop(r2,:)-pop(r3,:));
            vi = boundConstraint(vi, lu, budget);

            mask = rand(popsize, n) > CR;
            jrand = randi(n, popsize, 1);
            for i = 1:popsize
                mask(i, jrand(i)) = false;
            end
            ui = vi;
            ui(mask) = pop(mask);

            valOff = zeros(popsize, 1);
            for i = 1:popsize
                valOff(i) = obj(ui(i,:));
            end

            improved = valOff < val;
            pop(improved,:) = ui(improved,:);
            val(improved) = valOff(improved);

            [cur_min, idx] = min(val);
            best_x = pop(idx,:);
            real_val = sum(a .* (1 - exp(-b .* best_x(:))));
            best_curve(gen) = real_val;

            if gen == 50
                diversity50 = std(val);
            end
        end

        curves(run, :) = best_curve;
        final_vals(cr_idx, run) = best_curve(end);
        converge_iters(cr_idx, run) = find(best_curve > 0.99 * best_curve(end), 1);
        fprintf('CR=%.1f Run=%d Best=[%.2f %.2f %.2f %.2f %.2f], Conv=%.2f, D50=%.2f\n', ...
            CR, run, best_x, best_curve(end), diversity50);
    end

    mean_curves(cr_idx, :) = mean(curves);
end

% ========== 绘制对比收敛曲线 ==========
figure;
hold on;
colororder(lines(length(CR_list)))
for i = 1:length(CR_list)
    plot(1:maxGen, mean_curves(i,:), 'LineWidth', 1.8);
end
xlabel('迭代次数');
ylabel('平均最优目标函数值');
title('不同交叉率CR下的收敛曲线');
legend(arrayfun(@(cr) sprintf('CR=%.1f', cr), CR_list, 'UniformOutput', false));
grid on;
hold off;

% ========== 输出统计指标 ==========
mean_final = mean(final_vals, 2);
std_final = std(final_vals, 0, 2);
mean_conv = mean(converge_iters, 2);

fprintf('\nCR值\t均值\t\t标准差\t平均收敛代数\n');
for i = 1:length(CR_list)
    fprintf('%.1f\t%.2f\t%.2f\t%.2f\n', CR_list(i), mean_final(i), std_final(i), mean_conv(i));
end

% ========== 函数定义区 ==========
function f = obj(x)
    a = [10000;12000;8000;9000;15000];
    b = [0.5;0.4;0.6;0.5;0.3];
    lambda1 = 10000;
    lambda2 = 10000;
    budget = 20;
    conv = sum(a .* (1 - exp(-b .* x(:))));
    pen1 = lambda1 * max(0, sum(x) - budget)^2;
    pen2 = lambda2 * sum((max(0, 1 - x).^2 + max(0, x - 10).^2));
    f = -conv + pen1 + pen2;
end

function vi = boundConstraint(vi, lu, budget)
    xl = repmat(lu(1,:), size(vi,1),1);
    xu = repmat(lu(2,:), size(vi,1),1);
    vi = max(min(vi, 2.*xu - vi), 2.*xl - vi);
    vi = min(max(vi, xl), xu);
    for i = 1:size(vi, 1)
        sum_vi = sum(vi(i,:));
        if sum_vi > budget
            vi(i,:) = vi(i,:) * (budget / sum_vi);
        end
    end
end

function [r1, r2, r3] = rand3(N)
    r1 = zeros(N,1); r2 = zeros(N,1); r3 = zeros(N,1);
    for i = 1:N
        idx = randperm(N);
        r1(i) = idx(1);
        r2(i) = idx(2);
        r3(i) = idx(3);
    end
end

toc;