clear;
clc;
close all;

%% ============================================================
%  State-dependent dwell-time consensus simulation
%
%  MATLAB: R2024a
%  Solver: YALMIP + SeDuMi
%% ============================================================

fprintf("====================================================\n");
fprintf(" State-dependent dwell-time consensus simulation\n");
fprintf(" YALMIP and SeDuMi implementation\n");
fprintf("====================================================\n\n");

%% 0. Check YALMIP and SeDuMi

if exist("sdpvar", "file") ~= 2
    error("YALMIP is not found. Please add YALMIP to the MATLAB path.");
end

if exist("sedumi", "file") ~= 2
    error("SeDuMi is not found. Please add SeDuMi to the MATLAB path.");
end

fprintf("YALMIP path:\n%s\n\n", which("sdpvar"));
fprintf("SeDuMi path:\n%s\n\n", which("sedumi"));

yalmip('clear');

%% 1. Basic system parameters

N = 4;                  % Number of agents
d = 1;                  % Dimension of each agent state
numModes = 2;           % Number of candidate topologies

dt = 0.001;             % Simulation step
tf = 10;                % Final simulation time

%% 2. Candidate disconnected topologies

L = cell(numModes,1);

% The node arrangement used in the paper is
%
%       1 ----- 2
%       |       |
%       4 ----- 3
%
% Topology 1:
% reciprocal links 1 <-> 2 and 4 <-> 3
% disconnected components: {1,2} and {3,4}
L{1} = [ 1, -1,  0,  0;
        -1,  1,  0,  0;
         0,  0,  1, -1;
         0,  0, -1,  1];

% Topology 2:
% reciprocal links 1 <-> 4 and 2 <-> 3
% disconnected components: {1,4} and {2,3}
L{2} = [ 1,  0,  0, -1;
         0,  1, -1,  0;
         0, -1,  1,  0;
        -1,  0,  0,  1];

fprintf("L1 =\n");
disp(L{1});

fprintf("L2 =\n");
disp(L{2});

%% 3. Check the union graph

Lunion = L{1} + L{2};
eigUnion = sort(real(eig(Lunion)));

fprintf("Union Laplacian =\n");
disp(Lunion);

fprintf("Eigenvalues of the union Laplacian:\n");
disp(eigUnion.');

zeroTolerance = 1e-9;
numberOfZeroEigenvalues = sum(abs(eigUnion) < zeroTolerance);

if numberOfZeroEigenvalues ~= 1
    error("The union graph is not connected.");
end

fprintf("The union graph is connected.\n\n");

%% 4. Construct the consensus-orthogonal matrix S

S = [1/sqrt(2),  -1/sqrt(2),           0,            0;
     1/sqrt(6),   1/sqrt(6),  -2/sqrt(6),            0;
     1/sqrt(12),  1/sqrt(12),  1/sqrt(12), -3/sqrt(12)];

fprintf("S =\n");
disp(S);

residualS1 = norm(S*ones(N,1));
residualSS = norm(S*S' - eye(N-1), "fro");

fprintf("Residual of S times 1_N: %.4e\n", residualS1);
fprintf("Residual of S times S transpose: %.4e\n\n", residualSS);

if residualS1 > 1e-10
    error("The matrix S does not satisfy the consensus-orthogonality condition.");
end

if residualSS > 1e-10
    error("The rows of S are not orthonormal.");
end

Sd = kron(S,eye(d));
nxi = (N-1)*d;

%% 5. Construct the reduced-order disagreement matrices

Abar = cell(numModes,1);

for i = 1:numModes
    Abar{i} = -kron(S*L{i}*S',eye(d));

    fprintf("Reduced-order matrix Abar_%d =\n",i);
    disp(Abar{i});

    fprintf("Eigenvalues of Abar_%d:\n",i);
    disp(eig(Abar{i}).');
end

%% 6. Choose lambda and construct the convex combination

lambda = [0.5;0.5];

if any(lambda <= 0)
    error("All elements of lambda must be strictly positive.");
end

if abs(sum(lambda)-1) > 1e-12
    error("The elements of lambda must sum to one.");
end

Aaverage = zeros(nxi);

for i = 1:numModes
    Aaverage = Aaverage + lambda(i)*Abar{i};
end

eigAaverage = eig(Aaverage);
spectralAbscissaAverage = max(real(eigAaverage));

fprintf("\nlambda =\n");
disp(lambda);

fprintf("Convex combination of the reduced matrices =\n");
disp(Aaverage);

fprintf("Eigenvalues of the convex combination:\n");
disp(eigAaverage.');

fprintf("Spectral abscissa of the convex combination: %.12f\n", ...
    spectralAbscissaAverage);

if spectralAbscissaAverage >= -1e-10
    error("The convex combination is not Hurwitz.");
end

fprintf("The convex combination is Hurwitz.\n\n");

%% 7. Construct the Metzler matrix Pi

kappa = 0.54;

Pi0 = ones(numModes,1)*lambda' - eye(numModes);
Pi = kappa*Pi0;

fprintf("Pi0 =\n");
disp(Pi0);

fprintf("kappa = %.6f\n",kappa);

fprintf("Pi =\n");
disp(Pi);

rowSumResidual = norm(Pi*ones(numModes,1));
leftResidual = norm(lambda'*Pi);

fprintf("Residual of Pi times 1_M: %.4e\n",rowSumResidual);
fprintf("Residual of lambda transpose times Pi: %.4e\n",leftResidual);

if rowSumResidual > 1e-10
    error("Pi does not have zero row sums.");
end

if leftResidual > 1e-10
    error("Pi does not satisfy the left-invariance condition.");
end

offDiagonalMask = ~eye(numModes);

if any(Pi(offDiagonalMask) <= 0)
    error("The off-diagonal elements of Pi must be positive.");
end

fprintf("Pi is a valid row-sum-zero Metzler matrix.\n\n");

%% 8. Construct the zero-dwell-time lifted matrix

liftedDimension = nxi^2;
Bmode = cell(numModes,1);

for i = 1:numModes
    Bmode{i} = ...
        kron(eye(nxi),Abar{i}') ...
        + kron(Abar{i}',eye(nxi));
end

Bscript = blkdiag(Bmode{:});

J0 = Bscript + kron(Pi,eye(liftedDimension));

eigJ0 = eig(J0);
spectralAbscissaJ0 = max(real(eigJ0));

fprintf("Spectral abscissa of J0: %.12f\n", ...
    spectralAbscissaJ0);

if spectralAbscissaJ0 >= -1e-10
    error("The zero-dwell-time lifted matrix J0 is not Hurwitz.");
end

fprintf("The zero-dwell-time lifted matrix J0 is Hurwitz.\n\n");

%% 9. Solve the Lyapunov equation for the analytical bound

dimensionJ0 = size(J0,1);
identityJ0 = eye(dimensionJ0);

% Solve:
%
% J0 transpose times P + P times J0 = -I
%
% by vectorization, avoiding dependence on the Control System Toolbox.

lyapunovOperator = ...
    kron(eye(dimensionJ0),J0') ...
    + kron(J0',eye(dimensionJ0));

vectorP = -lyapunovOperator\identityJ0(:);

Pbound = reshape(vectorP,dimensionJ0,dimensionJ0);
Pbound = real((Pbound+Pbound')/2);

minEigenvaluePbound = min(real(eig(Pbound)));

if minEigenvaluePbound <= 0
    error("The Lyapunov matrix used for the dwell-time bound is not positive definite.");
end

PiDiagonal = diag(diag(Pi));
PiOffDiagonal = Pi-PiDiagonal;

normBscript = norm(Bscript,2);
normPbound = norm(Pbound,2);
normPiOffDiagonal = norm(PiOffDiagonal,2);

Tub = ...
    log(1 + 1/(2*normPbound*normPiOffDiagonal)) ...
    /normBscript;

fprintf("Norm of Bscript: %.12f\n",normBscript);
fprintf("Norm of Pbound: %.12f\n",normPbound);
fprintf("Norm of the off-diagonal part of Pi: %.12f\n", ...
    normPiOffDiagonal);

fprintf("Analytical dwell-time upper bound:\n");
fprintf("Tub = %.12f s\n\n",Tub);

%% 10. Select the prescribed dwell time

T = 0.1;

fprintf("Prescribed dwell time:\n");
fprintf("T = %.6f s\n",T);

if T >= Tub
    error("The prescribed dwell time must satisfy T < Tub.");
end

fprintf("The selected dwell time satisfies T < Tub.\n\n");

%% 11. Define the disagreement output matrix

C = eye(nxi);
Q = C'*C;

fprintf("C =\n");
disp(C);

fprintf("Q = C transpose times C =\n");
disp(Q);

%% 12. Compute the transition matrices and Y2

E = cell(numModes,1);
Y2 = cell(numModes,1);

for j = 1:numModes

    E{j} = expm(Abar{j}*T);

    integrand = @(tau) ...
        expm(Abar{j}'*tau)*Q*expm(Abar{j}*tau);

    Y2{j} = integral( ...
        integrand, ...
        0, ...
        T, ...
        "ArrayValued",true, ...
        "RelTol",1e-10, ...
        "AbsTol",1e-12);

    Y2{j} = real((Y2{j}+Y2{j}')/2);

    fprintf("E_%d =\n",j);
    disp(E{j});

    fprintf("Y2_%d =\n",j);
    disp(Y2{j});
end

%% ============================================================
%  13. Solve the Lyapunov--Metzler inequalities
%
%  X_i are full symmetric matrix decision variables.
%
%  For numerical conditioning, define:
%
%      X_i = scaleX times Z_i.
%
%  The L-M inequalities are divided by scaleX.
%% ============================================================

scaleX = 500;

fprintf("\nScaling factor: %.6f\n",scaleX);

Z = cell(numModes,1);

for i = 1:numModes
    Z{i} = sdpvar(nxi,nxi,'symmetric');
end

constraints = [];
LMscaled = cell(numModes,1);

positiveMargin = 1e-5;
negativeMargin = 1e-6;

for i = 1:numModes

    constraints = [
        constraints, ...
        Z{i} >= positiveMargin*eye(nxi)
    ];

end

for i = 1:numModes

    couplingScaled = zeros(nxi);

    for j = 1:numModes

        if j ~= i

            couplingScaled = couplingScaled ...
                + Pi(i,j)*( ...
                    E{j}'*Z{j}*E{j} ...
                    + Y2{j}/scaleX ...
                    - Z{i});

        end

    end

    LMscaled{i} = ...
          Abar{i}'*Z{i} ...
        + Z{i}*Abar{i} ...
        + couplingScaled ...
        + Q/scaleX;

    LMscaled{i} = ...
        (LMscaled{i}+LMscaled{i}')/2;

    constraints = [
        constraints, ...
        LMscaled{i} <= -negativeMargin*eye(nxi)
    ];

end

%% 14. Configure SeDuMi

options = sdpsettings( ...
    'solver','sedumi', ...
    'verbose',1, ...
    'warning',1, ...
    'sedumi.eps',1e-8);

%% 15. Solve the feasibility problem

fprintf("\n====================================================\n");
fprintf(" Solving the Lyapunov--Metzler inequalities\n");
fprintf("====================================================\n");

diagnostic = optimize(constraints,[],options);

fprintf("\nYALMIP diagnostic code: %d\n",diagnostic.problem);
fprintf("Diagnostic information: %s\n",diagnostic.info);
fprintf("YALMIP explanation: %s\n", ...
    yalmiperror(diagnostic.problem));

%% 16. Recover the Lyapunov matrices X_i

X = cell(numModes,1);
solverCandidateAvailable = true;

for i = 1:numModes

    Znumeric = value(Z{i});

    if isempty(Znumeric) || any(~isfinite(Znumeric(:)))
        solverCandidateAvailable = false;
        break;
    end

    X{i} = scaleX*Znumeric;
    X{i} = real((X{i}+X{i}')/2);

end

if ~solverCandidateAvailable
    error("The solver did not return valid Lyapunov matrices.");
end

if diagnostic.problem ~= 0
    fprintf("\nThe solver returned a warning or nonzero diagnostic code.\n");
    fprintf("The candidate solution will be checked using the original inequalities.\n");
end

%% 17. Compute Y1 using the obtained X_i

Y1 = cell(numModes,1);

for j = 1:numModes

    Y1{j} = E{j}'*X{j}*E{j};
    Y1{j} = real((Y1{j}+Y1{j}')/2);

end

%% 18. Verify the original unscaled L-M inequalities

LMnumeric = cell(numModes,1);

minEigX = zeros(numModes,1);
maxEigLM = zeros(numModes,1);

fprintf("\n====================================================\n");
fprintf(" Verification of the original L-M inequalities\n");
fprintf("====================================================\n");

for i = 1:numModes

    couplingOriginal = zeros(nxi);

    for j = 1:numModes

        if j ~= i

            couplingOriginal = couplingOriginal ...
                + Pi(i,j)*( ...
                    Y1{j} ...
                    + Y2{j} ...
                    - X{i});

        end

    end

    LMnumeric{i} = ...
          Abar{i}'*X{i} ...
        + X{i}*Abar{i} ...
        + couplingOriginal ...
        + Q;

    LMnumeric{i} = ...
        real((LMnumeric{i}+LMnumeric{i}')/2);

    eigXi = eig(X{i});
    eigLMi = eig(LMnumeric{i});

    minEigX(i) = min(real(eigXi));
    maxEigLM(i) = max(real(eigLMi));

    fprintf("\nMode %d\n",i);

    fprintf("X_%d =\n",i);
    disp(X{i});

    fprintf("Eigenvalues of X_%d:\n",i);
    disp(eigXi.');

    fprintf("Original L-M residual matrix for mode %d:\n",i);
    disp(LMnumeric{i});

    fprintf("Eigenvalues of the residual matrix:\n");
    disp(eigLMi.');

    fprintf("Minimum eigenvalue of X_%d: %.12e\n", ...
        i,minEigX(i));

    fprintf("Maximum eigenvalue of the L-M residual: %.12e\n", ...
        maxEigLM(i));

end

positiveTolerance = 1e-8;
negativeTolerance = 1e-8;

if any(minEigX <= positiveTolerance)
    error("At least one Lyapunov matrix is not positive definite.");
end

if any(maxEigLM >= -negativeTolerance)
    error("At least one original L-M inequality is not strictly satisfied.");
end

fprintf("\nAll X_i matrices are positive definite.\n");
fprintf("All original L-M residual matrices are negative definite.\n");
fprintf("The Lyapunov--Metzler inequalities are feasible.\n\n");

%% 19. Save the L-M solution

X1_numeric = X{1};
X2_numeric = X{2};

Y11_numeric = Y1{1};
Y12_numeric = Y1{2};

Y21_numeric = Y2{1};
Y22_numeric = Y2{2};

LM1_numeric = LMnumeric{1};
LM2_numeric = LMnumeric{2};

save("LM_solution_N4_T01.mat", ...
    "L", ...
    "Lunion", ...
    "S", ...
    "Abar", ...
    "lambda", ...
    "kappa", ...
    "Pi", ...
    "J0", ...
    "Pbound", ...
    "Tub", ...
    "T", ...
    "C", ...
    "Q", ...
    "X1_numeric", ...
    "X2_numeric", ...
    "Y11_numeric", ...
    "Y12_numeric", ...
    "Y21_numeric", ...
    "Y22_numeric", ...
    "LM1_numeric", ...
    "LM2_numeric", ...
    "minEigX", ...
    "maxEigLM", ...
    "diagnostic");

fprintf("The L-M solution was saved to LM_solution_N4_T01.mat.\n\n");

%% ============================================================
%  20. State-dependent dwell-time switching simulation
%% ============================================================

x = [5;-3;4;2];

xAverage = mean(x);

sigma = 1;
lastSwitchTime = 0;

% Include the initial time in the switching-time list
switchTimes = 0;

numberOfSteps = floor(tf/dt)+1;

tHistory = zeros(1,numberOfSteps);
xHistory = zeros(N*d,numberOfSteps);
xiHistory = zeros(nxi,numberOfSteps);

errorHistory = zeros(1,numberOfSteps);
rangeHistory = zeros(1,numberOfSteps);
sigmaHistory = zeros(1,numberOfSteps);

%% 21. Precompute exact state-transition matrices

stateTransition = cell(numModes,1);

for i = 1:numModes

    originalSystemMatrix = -kron(L{i},eye(d));

    stateTransition{i} = ...
        expm(originalSystemMatrix*dt);

end

%% 22. Main simulation loop

for k = 1:numberOfSteps

    currentTime = (k-1)*dt;
    xi = Sd*x;

    %% State-dependent switching condition

    if currentTime >= lastSwitchTime+T-1e-12

        switchRequired = false;
        selectedMode = sigma;
        minimumPredictedCost = inf;

        for j = 1:numModes

            if j == sigma
                continue;
            end

            switchingValue = ...
                xi'*(Y1{j}+Y2{j}-X{sigma})*xi;

            if switchingValue < -1e-10

                predictedCost = ...
                    xi'*(Y1{j}+Y2{j})*xi;

                if predictedCost < minimumPredictedCost

                    minimumPredictedCost = predictedCost;
                    selectedMode = j;
                    switchRequired = true;

                end

            end

        end

        if switchRequired

            sigma = selectedMode;
            lastSwitchTime = currentTime;

            switchTimes(end+1) = currentTime; %#ok<SAGROW>

        end

    end

    %% Store simulation data

    tHistory(k) = currentTime;
    xHistory(:,k) = x;
    xiHistory(:,k) = xi;

    errorHistory(k) = norm(xi,2);
    rangeHistory(k) = max(x)-min(x);
    sigmaHistory(k) = sigma;

    %% Exact state update

    if k < numberOfSteps
        x = stateTransition{sigma}*x;
    end

end

%% 23. Print the simulation results

fprintf("\n====================================================\n");
fprintf(" Simulation results\n");
fprintf("====================================================\n");

fprintf("Initial arithmetic average: %.8f\n",xAverage);

fprintf("Final agent states:\n");
disp(xHistory(:,end).');

fprintf("Final disagreement norm: %.8e\n", ...
    errorHistory(end));

fprintf("Final maximum pairwise difference: %.8e\n", ...
    rangeHistory(end));

fprintf("Expected consensus value: %.8f\n",xAverage);

if numel(switchTimes) >= 2

    dwellIntervals = diff(switchTimes);
    minimumObservedDwellTime = min(dwellIntervals);

    fprintf("Number of switches: %d\n", ...
        numel(switchTimes)-1);

    fprintf("Minimum observed switching interval: %.6f s\n", ...
        minimumObservedDwellTime);

    if minimumObservedDwellTime < T-1e-10
        warning("The prescribed minimum dwell-time constraint was violated.");
    else
        fprintf("The minimum dwell-time constraint is satisfied.\n");
    end

else

    fprintf("No switching occurred during the simulation.\n");

end

%% 24. Plot the agent states

figure;

plot(tHistory,xHistory(1,:),"LineWidth",1.5);
hold on;

plot(tHistory,xHistory(2,:),"LineWidth",1.5);
plot(tHistory,xHistory(3,:),"LineWidth",1.5);
plot(tHistory,xHistory(4,:),"LineWidth",1.5);

yline(xAverage,"--","LineWidth",1.2);

grid on;
box on;

xlabel("Time (s)");
ylabel("Agent states");

legend( ...
    "$x_1$", ...
    "$x_2$", ...
    "$x_3$", ...
    "$x_4$", ...
    "Initial average", ...
    "Interpreter","latex", ...
    "Location","best");


exportgraphics(gcf,"fig_state.png","Resolution",300);

%% 25. Plot the disagreement error

figure;

semilogy(tHistory,errorHistory,"LineWidth",1.5);
hold on;

semilogy(tHistory,rangeHistory,"--","LineWidth",1.5);

grid on;
box on;

xlabel("Time (s)");
ylabel("Disagreement error");

legend( ...
    "$\|\xi(t)\|_2$", ...
    "$\max_i x_i(t)-\min_i x_i(t)$", ...
    "Interpreter","latex", ...
    "Location","best");


exportgraphics(gcf,"fig_error.png","Resolution",300);

%% 26. Plot the switching signal

figure;

stairs(tHistory,sigmaHistory,"LineWidth",1.5);

grid on;
box on;

xlabel("Time (s)");
ylabel("\sigma(t)");

ylim([0.5,numModes+0.5]);
yticks(1:numModes);

exportgraphics(gcf,"fig_switching.png","Resolution",300);

%% 27. Print numerical values for the paper

fprintf("\n====================================================\n");
fprintf(" Numerical values for the paper\n");
fprintf("====================================================\n");

fprintf("lambda = [%.8f, %.8f] transpose\n", ...
    lambda(1),lambda(2));

fprintf("kappa = %.8f\n",kappa);
fprintf("T upper bound = %.8f s\n",Tub);
fprintf("T = %.8f s\n",T);

fprintf("Eigenvalues of the convex combination:\n");
disp(eigAaverage.');

fprintf("X1 =\n");
disp(X{1});

fprintf("X2 =\n");
disp(X{2});

fprintf("Maximum eigenvalue of LM1: %.12e\n", ...
    maxEigLM(1));

fprintf("Maximum eigenvalue of LM2: %.12e\n", ...
    maxEigLM(2));

fprintf("\nProgram finished.\n");