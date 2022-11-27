#NETWORK
#layers = [

#imageInputLayer([64, 64, 3]);

#convolution2dLayer([5, 5], 32, 'Padding','same');
#batchNormalizationLayer;
#reluLayer;
#maxPooling2dLayer(2, 'Stride', 2);

#convolution2dLayer([5, 5], 32, 'Padding','same');
#batchNormalizationLayer;
#reluLayer;
#maxPooling2dLayer(2, 'Stride', 2);

#convolution2dLayer([5, 5], 32, 'Padding','same');
#batchNormalizationLayer;
#reluLayer;
#maxPooling2dLayer(2, 'Stride', 2);

#fullyConnectedLayer(128);
#reluLayer;

#fullyConnectedLayer(2);
#softmaxLayer;
#classificationLayer;
#];

# TRAINING OPTINS
#options = trainingOptions('sgdm', ...
#    'MaxEpochs',40, ...
#    'ValidationData',imdsValidation, ...
#    'ValidationFrequency',30, ...
#    'Verbose',false, ...
#    'Plots','training-progress');


# TTRAIN YOUR NETWORK
#net = trainNetwork(imdsTrain,layers,options);