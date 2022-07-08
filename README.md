# climateHack

After trying many different strategies over 8 weeks, this was the model that I ultimately submitted for the 2022 Climate Hack competition. 
In this competition we were presented with 12 sateliite images of weather patterns (taken 5 minutes apart, representing one hour of data together) 
and asked to forecast the next 24 images (i.e. the next two hours) using machine learning models of our choice.
The model that I have implemented here consists of a 3d convolutional neural network followed by a fully connected layer (made with PyTorch). 
The idea is that the convolutonal layers extract features from the original time series and the fully connected layer reassembles these feautures
into a sensible prediction of the future. Though this model was not perfect, it scored a respectable 69% on the leaderboard and placed in the top 50 
overall. This was a great experience to partake in, much was learnt, so my thanks to everyone who organised it!
