function Offspring1=DEgenerator_better(Population1,Problem,FEA,epsilon)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao (email: qiaokangjia@yeah.net)

    p1           = Population1.decs;
    [popsize1,n] = size(p1);
    Offspring    = [];
    Fitness1     = CalFitness_E(Population1.objs,Population1.cons,epsilon);
    
    while length(Population1) > 1
        l = rand;
        if l <= 1/3
            F = .6;
        elseif l <= 2/3
            F = 0.8;
        else
            F = 1.0;
        end
        l = rand;
        if l <= 1/3
            CR = .1;
        elseif l <= 2/3
            CR = 0.2;
        else
            CR = 1.0;
        end
        indexset     = 1 : popsize1;
        r1           = floor(rand*(popsize1-1))+1;
        xr1          = indexset(r1);
        indexset(r1) = [];
        r2           = floor(rand*(popsize1-2))+1;
        xr2          = indexset(r2);
    
        if Fitness1(xr1) < Fitness1(xr2) %比较两个解
            best1  = xr1;
            worst1 = xr2;
        else
            best1  = xr2;
            worst1 = xr1;
        end
    
        v = p1(best1,:) + F*(p1(best1,:)-p1(worst1,:));
    
        o1    = rand(1,Problem.D); %随机产生替换的概率
        index = find(o1 <= 0.5);
    
        p1(worst1,index) = p1(best1,index); %差的解换好的解
        v2 = p1(worst1,:);
    
        %% Binomial crossover
        t      = rand(1, n) < CR;  % 按交叉概率CR生成逻辑向量（true=交叉）
        j_rand = floor(rand * n) + 1;  % 随机选1个维度（强制交叉，避免子代与父代一致）
        t(1, j_rand) = 1;  
        t_     = 1 - t;  % 反向逻辑向量
    
        v = t .* v + t_ .* p1(best1,:);  % 交叉生成新向量v（变异向量v与最优个体交叉）
    
        vv1      = p1(best1,:);
        vv1(FEA) = v(FEA);
        vv2      = p1(worst1,:);
        vv2(FEA) = v2(FEA);
    
        Offspring = [Offspring;vv1;vv2];
        Population1(:,[xr1,xr2]) = [];
        p1 = Population1.decs;
        [popsize1,n] = size(p1);  % 更新父代规模和维度
    end

    %% Polynomial mutation  多项式变异（维持种群多样性）
    [proM,disM] = deal(1,20);% proM：个体变异概率（1=每个个体都可能变异）；disM：分布指数（控制变异局部性）
    Lower       = repmat(Problem.lower,Problem.N,1); % 扩展下界到种群规模×维度
    Upper       = repmat(Problem.upper,Problem.N,1); % 扩展上界到种群规模×维度
    Site        = rand(Problem.N,Problem.D) < proM/Problem.D; % 变异位点（每个维度的变异概率=proM/D）
    mu          = rand(Problem.N,Problem.D);% 生成[0,1)随机数，控制变异幅度
    temp        = Site & mu<=0.5;% 第一种变异情况（mu≤0.5）
    Offspring       = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                      (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                      (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    
    Offspring1 = Problem.Evaluation(Offspring);
end