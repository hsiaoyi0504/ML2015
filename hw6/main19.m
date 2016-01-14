Z = csvread('hw2_lssvm_all.dat');
X = Z(:, 1:end - 1);
Y = Z(:, end);
[N, D] = size(X);

N_TRAIN = 400;
N_TEST = N - N_TRAIN;
X_TRAIN = X(1:N_TRAIN, :);
X_TEST = X(N_TRAIN+1:end, :);
Y_TRAIN = Y(1:N_TRAIN);
Y_TEST = Y(N_TRAIN+1:end);

GAMMA = 2 .^ [5, 1, -3];
LAMBDA = 10 .^ [3, 0, -3];

EIN = zeros(length(GAMMA), length(LAMBDA));
EOUT = zeros(length(GAMMA), length(LAMBDA));
for i = 1:length(GAMMA)
    gamma = GAMMA(i);
    for j = 1:length(LAMBDA)
        lambda = LAMBDA(j);
        K_TRAIN = exp(-gamma * squareform(pdist(X_TRAIN)) .^ 2);
        K_TEST = exp(-gamma * pdist2(X_TRAIN, X_TEST) .^ 2);
        beta = (lambda * eye(N_TRAIN) + K_TRAIN) \ Y_TRAIN;
        EIN(i, j) = sum(sign(K_TRAIN' * beta) ~= Y_TRAIN) / N_TRAIN;
        EOUT(i, j) = sum(sign(K_TEST' * beta) ~= Y_TEST) / N_TEST;
    end
end
