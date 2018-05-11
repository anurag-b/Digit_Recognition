Before starting the code, change the directory to which the train and test dataset is stored in the code. After doing so just follow the prompts. Select the appropriate dimensionality reduction 1. PCA 2. LDA, technique and then enter the percentage of the feature reduction for PCA and number of features for LDA. Later, you will be prompted to select the kernel function. Select 1. Linear 2. Poly 3. RBF. That's it just wait for the result.

For CNN, install caffe using the online documentation and then follow the steps below - 
Type this in your terminal if you are using Linux - 

Load Data - 

cd $CAFFE_ROOT
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh

Once the data is loaded just run the below command from the terminal - 

cd $CAFFE_ROOT
./examples/mnist/train_lenet.sh
