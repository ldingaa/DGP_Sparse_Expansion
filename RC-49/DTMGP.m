
clear all; close all; clc;
%% DTMGP
%% Load Data
load data
%images=h5read('RC-49_64x64.h5','/images');
%labels=h5read('RC-49_64x64.h5','/labels');
%ind_train=h5read('RC-49_64x64.h5','/indx_train');
trainX = preprocess(images); 
trainY = labels;
%testX = preprocess(imgs(:,:,1:4000:36001)); 
ind_test =[27656       32921        3634       74117];
testY=[40 80 120 160 200 240 280 320 360]';
realX=preprocess(testX);
%realX=reshape(realX,[64 64 4]);
%% Settings
settings.latent_dim = 100;
%settings.num_labels = 10;
settings.batch_size = 32; settings.image_size = [64,64,1]; 
settings.lrD = 0.0004; settings.lrG = 0.0004; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 50000;
numnodes = 256;
%% Initialization
%% Generator
%paramsGen.latentW=dlarray(zeros(settings.latent_dim,1),'CB');
paramsGen.FCW0=dlarray(initializeGaussian([settings.latent_dim,1],.02),'CB');
paramsGen.FCb0 = dlarray(zeros(settings.latent_dim,1,'single'));
%paramsGen.logvar=dlarray(zeros(settings.latent_dim,1),'CB');

paramsGen.FCW1 = dlarray(...
    initializeGaussian([256,201],.02));
paramsGen.FClogvar1=dlarray(-100*ones([256,201]));
paramsGen.FCb1 = dlarray(zeros(256,1,'single'));
%paramsGen.EMW1 = dlarray(...
%    initializeUniform([settings.latent_dim,...
%    settings.num_labels]));
%paramsGen.EMb1 = dlarray(zeros(1,settings.num_labels,'single'));
paramsGen.BNo1 = dlarray(zeros(256,1,'single'));
paramsGen.BNs1 = dlarray(ones(256,1,'single'));
paramsGen.FCW2 = dlarray(initializeGaussian([numnodes*2,513]));
paramsGen.FClogvar2 = dlarray(-100*ones([numnodes*2,513]));
paramsGen.FCb2 = dlarray(zeros(2*numnodes,1,'single'));
paramsGen.BNo2 = dlarray(zeros(2*numnodes,1,'single'));
paramsGen.BNs2 = dlarray(ones(2*numnodes,1,'single'));
paramsGen.FCW3 = dlarray(initializeGaussian([numnodes*4,1025]));
paramsGen.FCb3 = dlarray(zeros(4*numnodes,1,'single'));
paramsGen.BNo3 = dlarray(zeros(4*numnodes,1,'single'));
paramsGen.BNs3 = dlarray(ones(4*numnodes,1,'single'));
paramsGen.FCW4 = dlarray(initializeGaussian(...
    [prod(settings.image_size),4*numnodes]));
paramsGen.FCb4 = dlarray(zeros(prod(settings.image_size)...
    ,1,'single'));

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
        noise = gpdl(randn([settings.latent_dim,...
            settings.batch_size]),'CB');
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
            I{itr}=progressplot(paramsGen,stGen,settings,testY,realX);
            [~,~,loss(itr)] = ...
                dlfeval(@modelGradients,XBatch,YBatch,paramsGen,stGen,...
                settings);
            save I I
            save loss loss
            itr=itr+1;
        end
        
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s. ELBO="+extractdata(loss(itr-1)))
    epoch = epoch+1;
    data_DTMGP{epoch,1}=paramsGen;
    data_DTMGP{epoch,2}=stGen;
    save data_DTMGP data_DTMGP
    if epoch == settings.maxepochs
        out = true;
    end    
end
%% Helper Functions
%% preprocess
function x = preprocess(x)
x = double(x)/255;
x = (x-.5)/.5;
x = reshape(x,64*64,[]);
end
%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = (dlarray(x,labels));
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
function [dly,st] = Generator(labels,params,st)
[batch_num,~]=size(labels);
std=batch_num^0.5;
load R1
load R2
load R3
load SG1
load SG2
load SG3
%embedding
dly=fullyconnect(labels,params.FCW0,params.FCb0);
dly=leakyrelu(dly,0.2);

%1
dly=ker(SG1,dly);
dly=fullyconnect(dly,R1',zeros(201,1));
weight=params.FCW1+dlarray(randn(256,201,'single')/std).*exp(params.FClogvar1);
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
dly=fullyconnect(dly,R2',zeros(513,1));
weight=params.FCW2+dlarray(randn(512,513,'single')/std).*exp(params.FClogvar2);
dly = fullyconnect(dly,weight,params.FCb2);
%dly = leakyrelu(dly,0.2);
if isempty(st.BN2)
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,...
        params.BNo2,params.BNs2,'MeanDecay',.8);
else
    [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
        params.BNs2,st.BN2.mu,st.BN2.sig,...
        'MeanDecay',.8);
end
%3
dly=ker(SG3,dly);
dly=fullyconnect(dly,R3',zeros(1025,1));
dly = fullyconnect(dly,params.FCW3,params.FCb3);
%dly = leakyrelu(dly,0.2);
if isempty(st.BN3)
    [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,...
        params.BNo3,params.BNs3,'MeanDecay',.8);
else
    [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,...
        params.BNs3,st.BN3.mu,st.BN3.sig,...
        'MeanDecay',.8);
end
%4
dly = fullyconnect(dly,params.FCW4,params.FCb4);
% tanh
dly = tanh(dly);
end
%% modelGradients
function [GradGen,stGen,loss]=modelGradients(x,y,paramsGen,stGen,settings)
xPred = Generator(y,paramsGen,stGen);
squares = 0.5*(xPred-x).^2;
reconstructionLoss  = sum(squares, [1,2,3]);
KL1 = -.5 * sum(sum(1 + paramsGen.FClogvar1 - paramsGen.FCW1.^2 - exp(paramsGen.FClogvar1)));
KL2 = -.5 * sum(sum(1 + paramsGen.FClogvar2 - paramsGen.FCW2.^2 - exp(paramsGen.FClogvar2)));

loss=mean(20*reconstructionLoss + KL1+KL2);

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(loss,paramsGen,'RetainData',true);

end
%% progressplot
function I=progressplot(paramsGen,stGen,settings,testY,testX)

labels = gpdl(testY,'B');
%testX=pagetranspose(testX);
gen_imgs = Generator(labels,paramsGen,stGen);
gen_imgs = pagetranspose(reshape(gen_imgs,64,64,[]));
%gen_imgs(:,:,5:8)=testX;
%size(gen_imgs)

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

I = imtile(gatext(gen_imgs));
I = rescale(I);
imagesc(I)
title("Generated Images")
colormap gray
drawnow;
end
%% embedding
function dly = embedding(dlx,labels,params)
% params EM W (latent_dim,num_labels)
%               / (img_elements,num_labels)
%           b (latent_dim,1) (ignore)
%               / (img_elements,1)
% 'what are you doing？buddy!
maskW = params.EMW1(:,labels+1);
maskb = params.EMb1(:,labels+1);
dly = dlx.*maskW;
% dly = dlx.*maskW+maskb;
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