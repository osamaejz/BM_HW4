close all
clear all
load('BM59D_HW3_DATA.mat')

% Use either X or Y in below two lines depending upon the data you want to use.
data_points = X(:, 1:3); % Extract the first 3 columns as features
labels = X(:, 4);       % Extract the last column as labels

% Separate data by class
class0_indices = find(labels == 0); % Indices of class C0
class1_indices = find(labels == 1); % Indices of class C1

% Split into training and test sets
train_C0 = class0_indices(1:60); % First 60 instances of class C0
test_C0 = class0_indices(61:end); % Remaining 40 instances of class C0

train_C1 = class1_indices(1:60); % First 60 instances of class C1
test_C1 = class1_indices(61:end); % Remaining 40 instances of class C1

% Combine training and test sets
train_indices = [train_C0; train_C1];
test_indices = [test_C0; test_C1];

train_features = data_points(train_indices, :); % Training features
train_labels = labels(train_indices);          % Training labels

valid_features = data_points(test_indices, :);  % Test features
valid_labels = labels(test_indices);            % Test labels

% Initialize results storage
results = [];

% Define k values
k_values = [1, 3, 5];

% Perform k-NN for each k and distance metric
for k = k_values
    % --- Euclidean Distance ---
    preds_euclidean = zeros(size(valid_labels));
    for i = 1:size(valid_features, 1)
        % Replicate the test feature row to match the training feature size
        test_row = repmat(valid_features(i, :), size(train_features, 1), 1);

        % Compute Euclidean distance
        distances = sqrt(sum((train_features - test_row).^2, 2));

        % Sort distances and find the k-nearest labels
        [~, sorted_indices] = sort(distances); 
        nearest_labels = train_labels(sorted_indices(1:k)); 
        preds_euclidean(i) = mode(nearest_labels);
    end
    acc_euclidean = mean(preds_euclidean == valid_labels);
    conf_matrix_euclidean = confusionmat(valid_labels, preds_euclidean);
    
    % --- Mahalanobis Distance ---
    preds_mahalanobis = zeros(size(valid_labels));
    S = cov(train_features); % Covariance matrix
    invS = inv(S); % Inverse covariance matrix
    for i = 1:size(valid_features, 1)
        % Replicate the test feature row to match the training feature size
        test_row = repmat(valid_features(i, :), size(train_features, 1), 1);

        % Compute Mahalanobis distance
        diff = train_features - test_row;
        distances = diag(diff * invS * diff');
        
        % Sort distances and find the k-nearest labels
        [~, sorted_indices] = sort(distances); 
        nearest_labels = train_labels(sorted_indices(1:k)); 
        preds_mahalanobis(i) = mode(nearest_labels);
    end
    acc_mahalanobis = mean(preds_mahalanobis == valid_labels);
    conf_matrix_mahalanobis = confusionmat(valid_labels, preds_mahalanobis);
    
    % Store results
    results = [results; 
               {k, 'Euclidean', acc_euclidean, conf_matrix_euclidean};
               {k, 'Mahalanobis', acc_mahalanobis, conf_matrix_mahalanobis}];
end

% Display results
disp('Results:');
disp('k         Distance       Accuracy');
for i = 1:size(results, 1)
    fprintf('%-8d %-12s %.4f\n', results{i, 1}, results{i, 2}, results{i, 3});
    fprintf('Confusion Matrix:\n');
    disp(results{i, 4});
end
