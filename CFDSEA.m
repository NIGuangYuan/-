classdef CFDSEA < ALGORITHM
% <2026> <multi/many> <real/integer> <large/none> <constrained>
% It uses the original DVCEA threshold-based variable classification.

    methods
        function main(Algorithm,Problem)
            %% Initialization
            Population = Problem.Initialization();

            probLower  = 1 / (2 * sqrt(Problem.D));
            probUpper  = 1 - probLower;
            probCenter = 0.50;
            eta        = (probUpper - probCenter) / sqrt(Problem.D);

            [~, C] = kmeans(Population.decs,5);
            [FEA, INFEA] = Variable_classification(Problem,Population,C);
            rho = 1;
            Prob = CFDSEA.initProbability(Problem.D,FEA,INFEA,rho,probLower,probUpper,probCenter);

            cons = Population.cons;
            cons(cons <= 0) = 0;
            conss = sum(cons,2);
            epsilon0 = max(conss);
            if epsilon0 == 0
                epsilon0 = 1;
            end
            Fitness = CalFitness_E(Population.objs,Population.cons,epsilon0);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                cp      = (-log(epsilon0)-6) / log(1-0.5);
                epsilon = epsilon0 * (1-Problem.FE/Problem.maxFE)^cp;

                B_mask    = rand(1,Problem.D) < Prob;
                FEA_dyn   = find(B_mask == 1);
                INFEA_dyn = find(B_mask == 0);

                if isempty(FEA_dyn)
                    FEA_dyn = randi(Problem.D);
                end
                if isempty(INFEA_dyn)
                    INFEA_dyn = randi(Problem.D);
                end

                Offspring1 = OperatorDE_pbest_1_main(Population,Problem.N,Problem,Fitness,FEA_dyn,0.1);

                CV_pop  = sum(max(0,Population.cons),2);
                CV_off1 = sum(max(0,Offspring1.cons),2);

                if mean(CV_off1) < mean(CV_pop)
                    Prob(FEA_dyn)   = min(probUpper, Prob(FEA_dyn) + eta);
                    Prob(INFEA_dyn) = max(probLower, Prob(INFEA_dyn) - eta);
                else
                    Prob(FEA_dyn)   = max(probLower, Prob(FEA_dyn) - eta);
                end

                [Population,~] = Improve_E_EnvironmentalSelection([Population,Offspring1],Problem.N,epsilon);
                Offspring2 = DEgenerator_better(Population,Problem,INFEA_dyn,epsilon);
                [Population,Fitness] = Improve_E_EnvironmentalSelection([Population,Offspring2],Problem.N,epsilon);
            end
        end
    end

    methods(Static, Access = private)
        function Prob = initProbability(D, FEA, INFEA, rho, probLower, probUpper, probCenter)
            Prob   = probCenter * ones(1,D);
            margin = (probUpper - probCenter) * min(max(rho,0),1);
            Prob(FEA)   = probCenter + margin;
            Prob(INFEA) = probCenter - margin;
            Prob = min(probUpper, max(probLower, Prob));
        end
    end
end
