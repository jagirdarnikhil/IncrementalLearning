%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%Learn++ code explain in detail%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function test_learn()
K = 3;

addpath('C:\Users\jagir\Dropbox\Research projects\Project_OnlineRF_NJ\Code\Learn++\IncrementalLearning\src');   % add the code path 

load ionosphere      % load the built in ionosphere data set
%load sampledata
u = unique(Y);      % get the number of unique classes 
%u = {-1;0};      % created cell array based on the data
labels = zeros(numel(Y), 1);
%%
% convert the string labels to numeric labels
for n = 1:numel(Y)
  for c = 1:numel(u)
    if u{c} == Y{n}
      labels(n) = c;
      break
    end
  end
end
%% bootstrap of the data
% shuffle the data
i = randperm(numel(Y));
data = X(i, : );
labels = labels(i);
clear Description X Y c i n u

cv = cvpartition(numel(labels),'k',K);
z = zeros(numel(labels),1);
%%
for k = 1:K-1
  z = z + (training(cv,k)>0);
end
%%
ts_idx = find(z == K - 1);
tr_idx = find(z ~= K - 1);

data_tr = data(tr_idx, :);
data_te = data(ts_idx, :);
labels_tr = labels(tr_idx);
labels_te = labels(ts_idx);

cv = cvpartition(numel(labels_tr),'k',K);

%%
for k = 1:K
  data_tr_cell{k} = data_tr(training(cv,k)==0, :);
  labels_tr_cell{k} = labels_tr(training(cv,k)==0);
end
clear K cv data labels z tr_idx ts_idx k data_tr labels_tr 
%%
model.type = 'CART';
net.base_classifier = model;
net.iterations = 5;
net.mclass = numel(unique(labels_te));

%% Input for learn function

Tk = net.iterations;              % number of classifiers to generate
K = length(data_tr_cell);           % number of data sets 
net.classifiers = cell(Tk*K, 1);  % cell array with total number of classifiers
net.beta = zeros(Tk*K, 1);        % beta will set the classifier weights
c_count = 0;              % keep track of the number of classifiers at each time
errs = zeros(Tk*K, 1);    % prediction errors on the test data set

%%
% run learn++ on the data 
for k = 1:K  
  
  % obtain the latest data set and initialize the weights over the
  % instances to form a uniform distribution
  data_tr_cell_k = data_tr_cell{k};
  labels_tr_cell_k = labels_tr_cell{k};
  D = ones(numel(labels_tr_cell_k), 1)/numel(labels_tr_cell_k);
  
  % original paper says to modify D if prior knowledge is available. we can
  % modify the distribution weights if we already have a classifier
  % ensemble.
  if k > 1
    predictions = classify_ensemble(net, data_tr_cell_k, labels_tr_cell_k, ...
      c_count);   % predict on the training data
    epsilon_kt = sum(D(predictions ~= labels_tr_cell_k)); % error on D
    beta_kt = epsilon_kt/(1-epsilon_kt);                % normalized error on D
    D(predictions == labels_tr_cell_k) = beta_kt * D(predictions == labels_tr_cell_k);
  end
  
  for t = 1:Tk
    % update the classifier count 
    c_count = c_count + 1;
    
    % step 1 - make sure we are working with a probability distribution.  
    D = D / sum(D);
    
    % step 2 - grab a random sample of data indices with replacement from
    % the probability distribution D
    index = randsample(1:numel(D), numel(D), true, D);
    
    % step 3 - generate a new classifier on the data sampled from D. 
    net.classifiers{c_count} = classifier_train(...
      net.base_classifier, ...
      data_tr_cell_k(index, :), ...
      labels_tr_cell_k(index));
    
    % step 4 - test the latest classifier on ALL of the data not just the
    % data sampled from D, and compute the error according to the
    % probability distribution. then compute beta
    y = classifier_test(net.classifiers{c_count}, data_tr_cell_k);
    epsilon_kt = sum(D(y ~= labels_tr_cell_k));
    net.beta(c_count) = epsilon_kt/(1-epsilon_kt);
    
    % step 5 - get the ensemble decision computed with c_count classifiers
    % in the ensemble. compute the error on the probability distribution on
    % the composite hypothesis. 
    predictions = classify_ensemble(net, data_tr_cell_k, labels_tr_cell_k, ...
      c_count);
    E_kt = sum(D(predictions ~= labels_tr_cell_k));
    if E_kt > 0.5
      % rather than remove remove existing classifier; null the result out
      % by forcing the loss to be equal to 1/2 which is the worst possible
      % loss. feel free to modify the code to go back an iteration. 
      E_kt = 0.5;   
    end
    
    % step 6 - compute the normalized error of the compsite hypothesis and
    % update the weights over the training instances in the kth batch. 
    Bkt = E_kt / (1 - E_kt);
    D(predictions == labels_tr_cell_k) = Bkt * D(predictions == labels_tr_cell_k);
    D = D / sum(D);
    
    % make some predictions on the testing data set. 
    [predictions,posterior] = classify_ensemble(net, data_test, ...
      labels_test, c_count);
    errs(c_count) = sum(predictions ~= labels_test)/numel(labels_test); 
  end
  
  
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUXILARY FUNCTIONS
function [predictions,posterior] = classify_ensemble(net, data, labels, lims)
n_experts = lims;
weights = log(1./net.beta(1:lims));
p = zeros(numel(labels), net.mclass);
for k = 1:n_experts
  y = classifier_test(net.classifiers{k}, data);
  
  % this is inefficient, but it does the job 
  for m = 1:numel(y)
    p(m,y(m)) = p(m,y(m)) + weights(k);
  end
end
[~,predictions] = max(p');
predictions = predictions';
posterior = p./repmat(sum(p,2),1,net.mclass);
%%




[net,errs] = learn(net, data_tr_cell, labels_tr_cell, data_te, labels_te);
plot(errs)