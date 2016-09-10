function [nn,er] = SAE_2hl_60000(dropoutFrac, weightPenalty, inputZeroFrac)
%% train a 100-100 hidden unit SDAE and use it to initialize a FFNN
%% with provided parameters dropoutFrac, weightPenalty, inputZeroFrac
%% for sample of size 60000(Full sample)

load mnist_uint8;

train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([784 100 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.nonSparsityPenalty = 0.1;     %加入不稀疏的惩罚，从0.1开始，慢慢增加到2
sae.ae{1}.dropoutFraction = dropoutFrac;
sae.ae{1}.weightPenaltyL2 = weightPenalty;
sae.ae{1}.inputZeroMaskedFraction   = inputZeroFrac;    %一般0.5最好


sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.nonSparsityPenalty = 0.1;
sae.ae{2}.dropoutFraction = dropoutFrac;
sae.ae{2}.weightPenaltyL2 = weightPenalty;
sae.ae{2}.inputZeroMaskedFraction   = inputZeroFrac;


opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);

opts.numepochs =   5;   %再trainnumepoch*4次
sae.ae{1}.nonSparsityPenalty = 0.8;    %慢慢增大nonSparsityPenalty
sae.ae{2}.nonSparsityPenalty = 0.8;   
sae = saetrain(sae, train_x, opts);
sae.ae{1}.nonSparsityPenalty = 1.5;
sae.ae{2}.nonSparsityPenalty = 1.5;
sae = saetrain(sae, train_x, opts);
sae.ae{1}.nonSparsityPenalty = 1.8;
sae.ae{2}.nonSparsityPenalty = 1.8;
sae = saetrain(sae, train_x, opts);
sae.ae{1}.nonSparsityPenalty = 2;
sae.ae{2}.nonSparsityPenalty = 2;
sae = saetrain(sae, train_x, opts);

visualize(sae.ae{1}.W{1}(:,2:end)');

% Use the SDAE to initialize a FFNN
nn = nnsetup([784 100 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.output = 'softmax';
%nn.nonSparsityPenalty = 2;     %加入不稀疏的惩罚
nn.dropoutFraction = dropoutFrac;
nn.weightPenaltyL2 = weightPenalty;
nn.inputZeroMaskedFraction   = inputZeroFrac;    %一般0.5最好

%add pretrained weights
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   100;
opts.batchsize = 100;
opts.plot = 1; 
nn = nntrain(nn, train_x, train_y, opts, test_x, test_y);
[er, bad] = nntest(nn, test_x, test_y);
end