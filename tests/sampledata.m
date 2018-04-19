clear
clc
close all

%load ionosphere;
%% saving X and variable in .mat format along with data

addpath('C:\Users\jagir\Dropbox\Research projects\Project_OnlineRF_NJ\Code\matidatasets'); 

data = xlsread('FULL_DATASET.xlsx')

X = data(:,[1:47])
Y = data(:,[48])
Y = num2cell(Y)

save sampledata.mat X Y data ;
