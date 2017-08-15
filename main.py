import Initialization
from keras.layers import Activation, Dropout, Dense
from keras.layers import Input, Lambda
from keras.models import Model

# let's assume MNIST->USPS task.
domain_adaptation_task = 'MNIST_to_USPS'   # USPS_to_MNIST is also another option.

# let's run the experiments when 1 target sample per calss is available in training.
# you can run the experiments for sample_per_class=1, ... , 7.
sample_per_class = 1

# Running the experiments for repetition 5. In the paper we reported the average acuracy.
# We run the experiments for repetition=0,...,9 and take the average
repetition = 2


# Creating embedding function. This corresponds to the function g in the paper.
# You may need to change the network parameters.
model1=Initialization.Create_Model()

# size of digits 16*16
img_rows, img_cols = 16, 16
input_shape = (img_rows, img_cols, 1)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)


# number of classes for digits classification
nb_classes = 10

# Loss = (1-alpha)Classification_Loss + (alpha)CSA
alpha = .25

# Having two streams. One for source and one for target.
processed_a = model1(input_a)
processed_b = model1(input_b)


# Creating the prediction function. This corresponds to h in the paper.
out1 = Dropout(0.5)(processed_a)
out1 = Dense(nb_classes)(out1)
out1 = Activation('softmax', name='classification')(out1)


distance = Lambda(Initialization.euclidean_distance, output_shape=Initialization.eucl_dist_output_shape, name='CSA')(
    [processed_a, processed_b])
model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
model.compile(loss={'classification': 'categorical_crossentropy', 'CSA': Initialization.contrastive_loss},
              optimizer='adadelta',
              loss_weights={'classification': 1 - alpha, 'CSA': alpha})





# let's create the positive and negative pairs using row data.
# pairs will be saved in ./pairs directory
sample_per_class=5
for repetition in range(10):
    Initialization.Create_Pairs(domain_adaptation_task,repetition,sample_per_class)
    Acc=Initialization.training_the_model(model,domain_adaptation_task,repetition,sample_per_class)

    print('Best accuracy for {} target sample per class and repetition {} is {}.'.format(sample_per_class,repetition,Acc ))









