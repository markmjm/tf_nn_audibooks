import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from audiobooks_data_reader import Audiobooks_Data_Reader

#
#Extract the data from csc
raw_csv_data = np.loadtxt('./Audiobooks-data.csv',delimiter=',')
unscaled_input_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]
#
#Shuffle the data prior to further processing to randomize time raw data was acquired
raw_data_shuffled_indices = np.arange(unscaled_input_all.shape[0])
np.random.shuffle(raw_data_shuffled_indices)
unscaled_input_all = unscaled_input_all[raw_data_shuffled_indices]
targets_all = targets_all[raw_data_shuffled_indices]
#
#balance the input
num_one_targets = int(np.sum(targets_all))  # these are the converted customers 1+0+1+0+0+0+0+1 is 3 converted customers
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] ==0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_input_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)
#
#Standarize and scale the input to improve nn performance
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
#
#Shuffle the data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
#
#Split the data into train, validation, test.   Using 80%, 10%, 10%
samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count
train_inputs = shuffled_inputs[:train_samples_count]
train_targets  = shuffled_targets[:train_samples_count]
validation_inputs = shuffled_inputs[train_samples_count : train_samples_count + validation_samples_count]
validation_targets = shuffled_targets[train_samples_count :train_samples_count + validation_samples_count]
test_inputs = shuffled_inputs[train_samples_count + validation_samples_count :]
test_targets = shuffled_targets[train_samples_count + validation_samples_count :]
print(np.sum(train_targets), train_samples_count, np.sum(train_targets/train_samples_count))
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets/validation_samples_count))
print(np.sum(test_targets), test_samples_count, np.sum(test_targets/test_samples_count))
#
# Save data to npz files
np.savez('Audiobooks_data_train', inputs = train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs = validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs = test_inputs, targets=test_targets)
#
# Create the machine learning algorithem
# ## OUTLINE the model.
input_size =10
output_size = 2
hidden_layer_size = 50 # each hidden layer will have x nodes

tf.reset_default_graph() # clear anything fro previous run that is still in memory

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])  #(targets are 0, 1)

weights_1 = tf.get_variable('weights_1', [input_size, hidden_layer_size])
biases_1 = tf.get_variable('biases_1', [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable('weights_2', [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable('biases_2', [hidden_layer_size])
outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)


weights_3 = tf.get_variable('weights_3', [hidden_layer_size, output_size])
biases_3 = tf.get_variable('biases_3', [output_size])
outputs = tf.matmul(outputs_2, weights_3) + biases_3

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)

mean_loss = tf.reduce_mean(loss)

optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)

# ## Accuracy of prediction
out_equals_target = tf.equal(tf.argmax(outputs,1), tf.argmax(targets,1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

# ## Batching ans Stopping
sess = tf.InteractiveSession()
initializer = tf.global_variables_initializer()
sess.run(initializer)
batch_size = 100

max_epochs = 50
prev_validation_loss = 9999999.

train_data = Audiobooks_Data_Reader('train', batch_size)
validation_data = Audiobooks_Data_Reader('validation')

for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0.0

    for input_batch, target_batch in train_data:  # train data is an instance of Audiobooks_Data_Reader which is an iterable
        _, batch_loss = sess.run([optimize, mean_loss],feed_dict={inputs: input_batch, targets: target_batch})
        curr_epoch_loss += batch_loss

    curr_epoch_loss /= train_data.batch_count

    validation_loss = 0.0
    validation_accuracy = 0.0
    for input_batch, target_batch in validation_data:  # train validation_data is an instance of Audiobooks_Data_Reader which is an iterable
        validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],feed_dict={inputs: input_batch, targets: target_batch})

    print('Epoch ' + str(epoch_counter+1) +
          '. Training loss: ' + '{0:.3f}'.format(curr_epoch_loss) +
          '. Validation loss: ' + '{0:.3f}'.format(validation_loss) +
          '. Validation accuracy: ' + '{0:.3f}'.format(validation_accuracy * 100) + '%'
          )
    if validation_loss > prev_validation_loss:
        break
    prev_validation_loss = validation_loss
print('End of Training')
#
# TEST The model
test_data = Audiobooks_Data_Reader('test')
for input_batch, target_batch in test_data:  # train test_data is an instance of Audiobooks_Data_Reader which is an iterable
    test_accuracy = sess.run([accuracy], feed_dict={inputs: input_batch, targets: target_batch})
test_accuracy_percent = test_accuracy[0]*100.0
print(f'Test Accuracy: {test_accuracy_percent}')
