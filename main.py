import Initialization
from tensorflow.compat.v2.keras.layers import Activation, Dropout, Dense
from tensorflow.compat.v2.keras.layers import Input, Lambda
from tensorflow.compat.v2.keras.models import Model

# let's assume MNIST->USPS task.
domain_adaptation_task = 'MNIST_to_USPS'   # USPS_to_MNIST is also another option.

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




print('Domain Adaptation Task: ' + domain_adaptation_task)
# let's create the positive and negative pairs using row data.
# pairs will be saved in ./pairs directory
for sample_per_class in [1,3,5,7]:
    for repetition in range(10):
        model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
        model.compile(loss={'classification': 'categorical_crossentropy', 'CSA': Initialization.contrastive_loss},
                    optimizer='adadelta',
                    loss_weights={'classification': 1 - alpha, 'CSA': alpha})
        Initialization.Create_Pairs(domain_adaptation_task,repetition,sample_per_class)
        Acc=Initialization.training_the_model(model,domain_adaptation_task,repetition,sample_per_class)

        print('Best accuracy for {} target sample per class and repetition {} is {}.'.format(sample_per_class,repetition,Acc ))









