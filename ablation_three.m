%% Mesh Plot

filename = 'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\i_a_s\grit_hypertune_results.xlsx';
[Result, ~, raw] = xlsread(filename, 'val');

% Read table for easier column handling
T = readtable(filename, 'Sheet', 'val');

% Compute average score of Recall@10 and MRR@10
T.avgScore = (T.Recall_10 + T.MRR_10) / 2;

% Get unique dropout and attn_dropout values
dropout_vals = unique(T.dropout);
attn_vals = unique(T.attn_dropout);

% Initialize matrix for Z values
resultFinal = nan(length(attn_vals), length(dropout_vals));

% Fill matrix with average scores
for i = 1:length(attn_vals)
    for j = 1:length(dropout_vals)
        subset = T(T.attn_dropout == attn_vals(i) & T.dropout == dropout_vals(j), :);
        if ~isempty(subset)
            resultFinal(i, j) = mean(subset.avgScore);
        end
    end
end

% Create meshgrid for plotting
[X, Y] = meshgrid(1:length(dropout_vals), 1:length(attn_vals));

% Plot the surface
figure;
surf(X, Y, resultFinal);

% Set ticks/labels
set(gca, 'XTick', 1:length(dropout_vals), 'XTickLabel', string(dropout_vals));
set(gca, 'YTick', 1:length(attn_vals), 'YTickLabel', string(attn_vals));

% Font settings for axes
set(gca, 'FontName', 'Times New Roman', 'FontSize', 20, 'FontWeight', 'bold');
colormap('pink');
% Axis labels and title
xlabel('Dropout', 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 20);
ylabel('Attention Dropout', 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 20);
zlabel('Score', 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 20);
% title('Mesh Plot of Dropout vs Attention Dropout', 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 15);
box on;


% %% Contour Plot
% 
% filename = 'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\c_a_v\grit_hypertune_results.xlsx';
% 
% % Read table
% T = readtable(filename, 'Sheet', 'val');
% 
% % Compute average score of Recall@10 and MRR@10
% T.avgScore = (T.Recall_10 + T.MRR_10) / 2;
% 
% % Get unique num_groups and beta values
% groups_vals = unique(T.num_groups);
% beta_vals   = unique(T.beta);
% 
% % Initialize result matrix
% resultFinal = nan(length(groups_vals), length(beta_vals));
% 
% % Fill matrix with mean values
% for i = 1:length(groups_vals)
%     for j = 1:length(beta_vals)
%         subset = T(T.num_groups == groups_vals(i) & T.beta == beta_vals(j), :);
%         if ~isempty(subset)
%             resultFinal(i, j) = mean(subset.avgScore);
%         end
%     end
% end
% 
% % Create meshgrid
% [X, Y] = meshgrid(beta_vals, groups_vals);
% 
% % Use indices for equal spacing, but label with actual values
% contourf(1:length(beta_vals), 1:length(groups_vals), resultFinal, 20, 'LineColor', 'w');
% hcb = colorbar;
% 
% % Set ticks to actual values
% set(gca, 'XTick', 1:length(beta_vals), 'XTickLabel', string(beta_vals));
% set(gca, 'YTick', 1:length(groups_vals), 'YTickLabel', string(groups_vals));
% 
% % Colormap (inverse bone)
% colormap(bone);
% 
% % Font settings
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 15);
% set(hcb, 'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight', 'bold');
% 
% % Axis labels
% xlabel('\beta', 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 15);
% ylabel('\kappa', 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 15);
% 
% % Optional title
% % title('Contour Plot of Avg(Recall@10, MRR@10)', ...
% %     'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 15);
% 
% %% Bar Plot
% % List of dataset files
% files = {
%     'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\c_a_v\grit_hypertune_results.xlsx';
%     'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\i_a_s\grit_hypertune_results.xlsx';
%     'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\ml-1m\grit_hypertune_results.xlsx';
%     'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\ml-100k\grit_hypertune_results.xlsx';
%     'D:\Working\9. GrIT\Codes\experiments\hyperparameter_tuning\v_g\grit_hypertune_results.xlsx';
% };
% 
% datasets = cellfun(@(f) erase(extractAfter(f, 'results_'), '.xlsx'), files, 'UniformOutput', false);
% 
% % Initialize storage
% allGroups = [];
% allBetas  = [];
% 
% % First, collect all unique groups and betas across datasets
% for f = 1:length(files)
%     T = readtable(files{f}, 'Sheet', 'val');
%     T.avgScore = (T.Recall_10 + T.MRR_10) / 2;
%     allGroups = union(allGroups, unique(T.num_groups));
%     allBetas  = union(allBetas, unique(T.beta));
% end
% 
% % Matrices: rows = datasets, cols = unique groups/betas
% groupMat = nan(length(datasets), length(allGroups));
% betaMat  = nan(length(datasets), length(allBetas));
% 
% % Fill matrices
% for f = 1:length(files)
%     T = readtable(files{f}, 'Sheet', 'val');
%     T.avgScore = (T.Recall_10 + T.MRR_10) / 2;
% 
%     % groups
%     for j = 1:length(allGroups)
%         idx = T.num_groups == allGroups(j);
%         if any(idx)
%             groupMat(f, j) = mean(T.avgScore(idx));
%         end
%     end
% 
%     % betas
%     for j = 1:length(allBetas)
%         idx = T.beta == allBetas(j);
%         if any(idx)
%             betaMat(f, j) = mean(T.avgScore(idx));
%         end
%     end
% end
% % Colors (from your Python palette)
% palette = [
%     1.0, 1.0, 0.31;   % Lemon Yellow "#FFF44F"
%     1.0, 0.44, 0.57;  % Blush Pink "#FF6F91"
%     0.53, 0.81, 0.92; % Sky Blue "#87CEEB"
%     0.56, 0.93, 0.56; % Light Green "#8EEA8E"
%     1.0, 0.64, 0.0;   % Orange "#FFA300"
%     0.80, 0.60, 0.78; % Lavender "#CC99C9"
%     0.98, 0.80, 0.69; % Peach "#FACCA0"
%     0.67, 0.84, 0.90; % Light Cyan "#AAD5E6"
% ];
% 
% % Dataset display names
% datasetMap = containers.Map( ...
%     {'ml_1m','ml_100k','v_g','i_a_s','c_a_v'}, ...
%     {'MovieLens 1M','MovieLens 100K','Video Games','Industrial & Scientific','CDs & Vinyl'} ...
% );
% datasetKeys = {'ml_1m','ml_100k','v_g','i_a_s','c_a_v'};
% datasetNames = cellfun(@(k) datasetMap(k), datasetKeys, 'UniformOutput', false);
% 
% % Function to plot grouped bars
% function plotGroupedBars(mat, datasetNames, paramValues, xlabelText, titleText, palette)
%     figure;
%     b = bar(mat, 'grouped');  % grouped bars
% 
%     % Set colors
%     for j = 1:length(b)
%         b(j).FaceColor = palette(j,:);
%         b(j).EdgeColor = 'k';
%         b(j).LineWidth = 0.5;
%     end
% 
%     % Axes formatting
%     set(gca, 'XTick', 1:length(datasetNames), 'XTickLabel', datasetNames, ...
%         'FontName', 'Times New Roman', 'FontSize', 12);
%     xlabel(xlabelText, 'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight','bold');
%     ylabel('Score', 'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight','bold');
% 
%     % Grid
%     grid on;
%     ax = gca;
%     ax.YGrid = 'on';
%     ax.XGrid = 'off';
%     ax.GridLineStyle = '--';
%     ax.GridAlpha = 0.5;
% 
%     % Legend above plot
%     legend(string(paramValues), 'Location','northoutside','Orientation','horizontal', ...
%         'FontName','Times New Roman', 'FontSize', 10, 'Box','off');
% 
%     % title(titleText, 'FontName','Times New Roman', 'FontSize',12,'FontWeight','bold');
% end
% 
% % Plot groups (κ)
% plotGroupedBars(groupMat, datasetNames, allGroups, 'Dataset', 'Effect of \kappa across datasets', palette);
% 
% % Plot betas (β)
% plotGroupedBars(betaMat, datasetNames, allBetas, 'Dataset', 'Effect of \beta across datasets', palette);
