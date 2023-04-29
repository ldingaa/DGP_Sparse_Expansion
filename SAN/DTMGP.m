
clear all; close all; clc;
%% DTMGP
%% generate data Data
load data_lhd_100000.mat
Y_train=(Y_train-l)/(u-l);
Y_test=(Y_test-l)/(u-l).*.9955;

N=50000;
trainX = X_train(:,1:N); 
trainY = Y_train(1:N);
%testX=.5*ones(2,100);

%% Settings
settings.batch_size = 100; 
settings.lrD = 0.0002; settings.lrG = 0.0001; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 20000;

%% Initialization
%% Generator
%paramsGen.latentW=dlarray(zeros(settings.latent_dim,1),'CB');
%paramsGen.mean=dlarray(zeros(settings.latent_dim,1),'CB');
%paramsGen.logvar=dlarray(zeros(settings.latent_dim,1),'CB');

paramsGen.FCW1 = dlarray(initializeGaussian([10,391],.02));
paramsGen.FClogvar1=dlarray(-2.5*ones([10,391]));
paramsGen.FCb1 = dlarray(zeros(10,1,'single'));

paramsGen.BNo1 = dlarray(zeros(10,1,'single'));
paramsGen.BNs1 = dlarray(ones(10,1,'single'));


paramsGen.FCW2 = dlarray(initializeGaussian([1,241]));
%paramsGen.FClogvar2=dlarray(-ones([1,241]));
paramsGen.FCb2 = dlarray(zeros(1,1,'single'));
%paramsGen.BNo2 = dlarray(zeros(1,1,'single'));
%paramsGen.BNs2 = dlarray(ones(1,1,'single'));


stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = [];

% average Gradient and average Gradient squared holders
 avgG.Gen = []; avgGS.Gen = [];
%{
load paramsGen
load avgGS
load avgG
load stGen
load GradGen
%}

 %{
save paramsGen paramsGen
save avgGS avgGS
save avgG avgG
save stGen stGen
save GradGen  GradGen
 %}
%% Train
numIterations = floor(size(trainX,2)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
%}
itr=1;
while ~out
    tic; 
    shuffleid = randperm(size(trainX,2));
    trainXshuffle = trainX(:,shuffleid);
    trainYshuffle = trainY(shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch=gpdl(single(trainXshuffle(:,idx)),'CB');
        YBatch=gpdl(single(trainYshuffle(idx)),'B');

        [GradGen,stGen,~] = ...
                dlfeval(@modelGradients,XBatch,YBatch,paramsGen,stGen,...
                settings);

        % Update Generator network parameters
         
        [paramsGen,avgGS.Gen] = rmspropupdate(...
                  paramsGen,GradGen,avgGS.Gen,settings.lrG);
      %{ 
        [paramsGen,avgG.Gen,avgGS.Gen] = ...
            adamupdate(paramsGen, GradGen, ...
            avgG.Gen, avgGS.Gen, global_iter, ...
            settings.lrG, settings.beta1, settings.beta2);
       %} 
        if i==1 || rem(i,100)==0
            trueY=Y_test;
            testY=zeros(100,100);
            for t=1:100
                testY(:,t)=extractdata(Generator(gpdl(single(X_test),'CB'),paramsGen,stGen));
            end
            
            KSSTAT=zeros(1,100);
            for t=1:100
                [~,~,KSSTAT(t)]=kstest2(testY(t,:),trueY(t,:));
            end
                
                s(itr)=mean(KSSTAT);
                %settings.lrG=exp(-1/(s(end)));
                %min([.9,exp(-1/(2*s(end)))])
                [~,~,loss(itr)] = ...
                dlfeval(@modelGradients,XBatch,YBatch,paramsGen,stGen,...
                settings);
                
                subplot(2,1,1)
                plot(1:itr,s)
                %surf(reshape(testY,[50 50]))
                title('KSTAT')
                drawnow
            
                subplot(2,1,2)
                plot(1:itr, loss)
                title('ELBO')
                drawnow
               
                itr=itr+1;
        end
        
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s. ELBO="+extractdata(loss(itr-1)))
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end
%% Helper Functions
%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end
%% Weight initialization
function parameter = initializeGaussian(parameterSize,sigma)
if nargin < 2
    sigma = 0.05;
end
parameter = randn(parameterSize, 'single') .* sigma;
end
function parameter = initializeUniform(parameterSize,sigma)
if nargin < 2
    sigma = 0.05;
end
parameter = 2*sigma*rand(parameterSize, 'single')-sigma;
end
%% Generator
function [dly,st] = Generator(dlx,params,st)
[batch_num,~]=size(dlx);
std=batch_num^0.5;
load R1
load R2
load SG1
load SG2

%embedding


%1
dly=ker(SG1,dlx);
dly=fullyconnect(dly,R1',zeros(391,1));
weight=params.FCW1+dlarray(randn(10,391,'single')/std).*exp(params.FClogvar1);

dly = fullyconnect(dly,weight,params.FCb1);
%dly = leakyrelu(dly,0.2);
if isempty(st.BN1)
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,...
        params.BNo1,params.BNs1,'MeanDecay',.8);
else
    [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
        params.BNs1,st.BN1.mu,st.BN1.sig,...
        'MeanDecay',.8);
end
%2
dly=ker(SG2,dly);
dly=fullyconnect(dly,R2',zeros(241,1));
%weight=params.FCW2+dlarray(randn(1,241,'single')/std).*exp(params.FClogvar2);
dly = fullyconnect(dly,params.FCW2,params.FCb2);
%dly = fullyconnect(dly,params.FCW2,params.FCb2);
%dly = leakyrelu(dly,0.2);
%dly = dropout(dly);
%{
 if isempty(st.BN2)
     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,params.BNs2);
 else
     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
         params.BNs2,st.BN2.mu,st.BN2.sig);
 end
%}
%3
%{
if isempty(st.BN2)
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,...
        params.BNo2,params.BNs2,'MeanDecay',.8);
else
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
        params.BNs2,st.BN2.mu,st.BN2.sig,...
        'MeanDecay',.8);
end
%}
% tanh
dly = sigmoid(dly);
end
%% modelGradients
function [GradGen,stGen,loss]=modelGradients(x,y,paramsGen,stGen,settings)
%{
m=3;
L=zeros(1,m);
for i=1:m 
L(i) = sum(abs(Generator(x,paramsGen,stGen)-y));
end
size(L)
reconstructionLoss  = mean(L,2);
%}
m=5;
for i=1:m
yPred = Generator(x,paramsGen,stGen);
squares(i,:) = (yPred-y).^2;
end

reconstructionLoss  = sum(squares,[1 2])/m;
v1=.05;
%v2=.05;
KL1 = .5 * sum(sum(-1 -paramsGen.FClogvar1+paramsGen.FCW1.^2+log(v1)+ exp(paramsGen.FClogvar1)/v1));
%KL2 = .5 * sum(sum(-1 -paramsGen.FClogvar2+paramsGen.FCW2.^2+log(v2)+ exp(paramsGen.FClogvar2)/v2));

%loss=mean(reconstructionLoss*1000 + KL1/settings.batch_size+KL2/settings.batch_size );
loss=mean(reconstructionLoss+ KL1/settings.batch_size);

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(loss,paramsGen,'RetainData',true);

end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = (dlarray(x,labels));
end
%% kernel function
function  S = ker(X1_scaled,X2_scaled)


n1 = size(X1_scaled,1);
d = size(X1_scaled,2);
n2 = size(X2_scaled,2);


X1_scaled = X1_scaled/d;
X2_scaled = X2_scaled/d;
F(1,:,:) = X2_scaled;
diff_val = abs(repmat(X1_scaled,[1,1,n2])-repmat(F,[n1,1,1]));

S = exp(-squeeze(sum(diff_val,2)));
S=dlarray(S,'SB');

end

%% dropout
function dly = dropout(dlx,p)
if nargin < 2
    p = .3;
end
n = p*10;
mask = randi([1,10],size(dlx));
mask(mask<=n)=0;
mask(mask>n)=1;
dly = dlx.*mask;

end
