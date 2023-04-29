
clear all; close all; clc;
%% DTMGP
%% generate data Data
N=100;
trainX = rand(2,N); 
trainY = RFgen(trainX);
%testX=.5*ones(2,100);

%% Settings
settings.batch_size = 32; 
settings.lrD = 0.0002; settings.lrG = 0.0002; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 2000;

%% Initialization
%% Generator
%paramsGen.latentW=dlarray(zeros(settings.latent_dim,1),'CB');
%paramsGen.mean=dlarray(zeros(settings.latent_dim,1),'CB');
%paramsGen.logvar=dlarray(zeros(settings.latent_dim,1),'CB');

paramsGen.FCW1 = dlarray(initializeGaussian([1,181],.02));
paramsGen.FClogvar1=dlarray(zeros([1,181]));
paramsGen.FCb1 = dlarray(zeros(1,1,'single'));

paramsGen.BNo1 = dlarray(zeros(1,1,'single'));
paramsGen.BNs1 = dlarray(ones(1,1,'single'));


paramsGen.FCW2 = dlarray(initializeGaussian([1,127]));
paramsGen.FCb2 = dlarray(zeros(1,1,'single'));


stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = [];

% average Gradient and average Gradient squared holders
 avgG.Gen = []; avgGS.Gen = [];

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
        [paramsGen,avgG.Gen,avgGS.Gen] = ...
            adamupdate(paramsGen, GradGen, ...
            avgG.Gen, avgGS.Gen, global_iter, ...
            settings.lrG, settings.beta1, settings.beta2);
        
        if i==1 || rem(i,100)==0
            testX=rand(2,100);
            testY=zeros(100,100);
            trueY=zeros(100,100);
            for t=1:100
                testY(:,t)=extractdata(Generator(gpdl(single(testX),'CB'),paramsGen,stGen));
                trueY(:,t)=RFgen(testX);
            end
            KSSTAT=zeros(1,100);
            for t=1:100
                [~,~,KSSTAT(t)]=kstest2(testY(t,:),trueY(t,:));
            end
                
                s(itr)=mean(KSSTAT);
                subplot(2,1,1)
                plot(1:itr,s)
                %surf(reshape(testY,[50 50]))
                title('KSTAT')
                drawnow
                [~,~,loss(itr)] = ...
                dlfeval(@modelGradients,XBatch,YBatch,paramsGen,stGen,...
                settings);
                subplot(2,1,2)
                plot(1:itr, loss)
                title('ELBO')
                drawnow
                itr=itr+1;
        end
        
    end

    elapsedTime = toc;
    %disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s. ELBO="+extractdata(loss(itr-1)))
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
load R1_new
load R2
load SG1
load SG2

%embedding


%1
dly=ker(SG1,dlx);
dly=fullyconnect(dly,R1',zeros(181,1));
weight=params.FCW1+dlarray(randn(1,181,'single')/std).*exp(params.FClogvar1);
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
dly=fullyconnect(dly,R2',zeros(127,1));

dly = fullyconnect(dly,params.FCW2,params.FCb2);

% tanh
%dly = tanh(dly);
end
%% modelGradients
function [GradGen,stGen,loss]=modelGradients(x,y,paramsGen,stGen,settings)
yPred = Generator(x,paramsGen,stGen);
squares = (yPred-y).^2;
reconstructionLoss  = sum(squares);
KL1 = .5 * sum(sum(-1 -paramsGen.FClogvar1 + paramsGen.FCW1.^2 + exp(paramsGen.FClogvar1)));


loss=mean(reconstructionLoss + KL1);

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

%% Generation from the true random field
function Y=RFgen(X)

[~,N]=size(X);

%generate the covariance matrix K of Brownian sheet
K=zeros(N);
for i=1:N
    for j=1:N
        K(i,j)= (min(X(1,i),X(1,j))+1)*(min(X(2,i),X(2,j))+1);
    end
end

%generate Brownian sheet at X
Y=mvnrnd(zeros(N,1),K);

%Apply the logistic function on $Y$
Y=1./(1+exp(-Y));


end
