# ###############################################################################################################################################
# 2.2 Define Training Module
def train_model(model, params):
    with strategy.scope():
        optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
        train_metric = metrics.MeanSquaredError(name='train_mse')
        valid_loss = metrics.Mean(name='valid_loss')
        valid_metric = metrics.MeanSquaredError(name='valid_mse')
        loss_func = CustomLoss(params['performance_metric'], int(params['global_batch_size'] / params['minibatch_size']), params['global_batch_size'])

    num_epochs = params['num_epochs']
    tf_data_train = params['training_dataset']
    hidden_layer_size = params['hidden_layer_size']
    max_norm = params['max_norm']

    def train_step(data): #, chunk_sizes
        inputs = data['inputs']
        outputs = data['outputs']
        active_entries = data['active_entries']
        weights = data['propensity_weights'] if 'propensity_weights' in data else tf.constant(1.0)

        with tf.GradientTape() as tape:
       
            batch_size = tf.shape(inputs)[0]
            initial_state = tf.zeros([batch_size, hidden_layer_size], dtype=tf.float32)
            predictions,_,_ = model([inputs,initial_state, initial_state], training=True)
            # Compute loss
            loss = loss_func.train_call(outputs, predictions, active_entries, weights)
            # loss = compute_mse_loss(outputs, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm = max_norm)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #self.train_loss.update_state(loss)
        train_metric.update_state(outputs, predictions)

        return loss

    @tf.function
    def distributed_train_step(data): #, chunk_sizes
        per_replica_losses = strategy.run(train_step, args=(data,)) #, chunk_sizes
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    for epoch in range(num_epochs):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in tf_data_train:
            total_loss += distributed_train_step(x)
            num_batches += 1
            train_loss = total_loss / num_batches

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                  "Test Accuracy: {}")
        print(template.format(epoch + 1, train_loss,
                             train_metric.result() * 100, valid_loss.result(),
                             valid_metric.result() * 100))

        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

class TrainModule(tf.Module):
    def __init__(self, model, params, name=None):
        super(TrainModule, self).__init__(name=name)
        #self.strategy = strategy
        with self.name_scope:  #相当于with tf.name_scope("demo_module")
            self.model = model
            self.epochs = params['num_epochs']
            self.ds_train = params['training_dataset']
            self.ds_valid = params['validation_dataset']
            self.ds_test = params['test_dataset']
            self.input_size = params['input_size']
            self.minibatch_size = params['minibatch_size']
            self.global_batch_size = params['global_batch_size']
            self.hidden_layer_size = params['hidden_layer_size']
            self.performance_metric = params['performance_metric']
            self.max_global_norm = params['max_norm']
            self.backprop_length = params['backprop_length']
            self.model_folder = params['model_folder']

            self.optimizer = params['optimizer']
            #self.loss_func = CustomLoss(self.performance_metric, int(self.global_batch_size / self.minibatch_size), self.global_batch_size)
            # Set up checkpoint directory
            self.checkpoint_dir = os.path.join(self.model_folder, 'training_checkpoints')
            # train_loss: Track the average loss throughout the training process 
            # by calculating the average of the loss values in all training steps.
            #self.train_loss = metrics.Mean(name='train_loss') 
            # train_metric: Track the average MSE throughout the training process
            self.train_metric = params['train_metric']
            self.valid_loss = params['valid_loss']
            self.valid_metric =params['valid_metric']


    def train_step(self, data): #, chunk_sizes
        inputs = data['inputs']
        outputs = data['outputs']
        active_entries = data['active_entries']
        weights = data['propensity_weights'] if 'propensity_weights' in data else tf.constant(1.0)

        with tf.GradientTape() as tape:
            #segment_predictions = []
            #start = 0
            #states = None
            #for i, chunk_size in enumerate(chunk_sizes):
            #    input_chunk = tf.slice(inputs, [0, start, 0], [-1, chunk_size, self.input_size])
            #    if states is None:
            #        batch_size = tf.shape(input_chunk)[0]
            #        initial_state = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
            #        segment_prediction, state_h, state_c = model([input_chunk, initial_state, initial_state], training=True)
            #    else:
            #        segment_prediction, state_h, state_c = model([input_chunk, states[0], states[1]], training=True)
           
            #    segment_predictions.append(segment_prediction)
            #    # break links between states for truncated bptt
            #    states = [state_h, state_c]
            #    #states = tf.identity(states)
            #    # Starting point
            #    start += chunk_size

            ## Dumping output
            #predictions = tf.concat(segment_predictions, axis=1)
            batch_size = tf.shape(inputs)[0]
            initial_state = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
            predictions,_,_ = self.model([inputs,initial_state, initial_state], training=True)
            # Compute loss
            #loss = self.loss_func.train_call(outputs, predictions, active_entries, weights)
            loss = compute_mse_loss(outputs, predictions)
        
            #predictions = model(inputs, training=True)
            #loss = self.loss_func.train_call(outputs, predictions, active_entries, weights)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm = self.max_global_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        #self.train_loss.update_state(loss)
        self.train_metric.update_state(outputs, predictions)

        return loss

    @tf.function
    def distributed_train_step(self, data): #, chunk_sizes
        per_replica_losses = strategy.run(self.train_step, args=(data,)) #, chunk_sizes
        tf.print("finish per replica losses.")
        tf.print(strategy.experimental_local_results(per_replica_losses)[0].device)
        #tf.print(strategy.experimental_local_results(per_replica_losses)[1].device)
        loss = strategy.reduce("SUM", per_replica_losses, axis=None)
        tf.print(f"loss={loss}")
        
        return loss
        #return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    #@tf.function
    def valid_step(self, inputs, outputs, active_entries):

        batch_size = tf.shape(inputs)[0]
        initial_state = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
        predictions, _, _ = self.model([inputs, initial_state, initial_state], training=False)
        loss = self.loss_func.valid_call(outputs, predictions)
        sample_weight = active_entries/tf.reduce_sum(active_entries)
    
        self.valid_loss.update_state(loss, sample_weight=sample_weight)
        self.valid_metric.update_state(outputs, predictions)

    @tf.function
    def distributed_valid_step(self, inputs, outputs, active_entries):
        strategy.run(self.valid_step, args=(inputs, outputs, active_entries))

    def train_model(self, params, 
                    use_truncated_bptt=True, 
                    b_stub_front=True,
                    b_use_state_initialisation=True):

        # Create a checkpoint
        #checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        # initialize history
        history = {
            'train_loss': [],
            'train_mse': [],
            'valid_loss': [],
            'valid_mse': []
            }
        min_epochs = 50
        min_loss = tf.constant(np.inf)
        for epoch in tf.range(1, self.epochs+1):
            total_loss = 0
            num_batches = 0
  
            for data in self.ds_train:
                
                #input_data = data['inputs']
                #output_data = data['outputs']
                #active_entries = data['active_entries']
                #weights = data['propensity_weights'] if 'propensity_weights' in data else tf.constant(1.0)

                # Stack up the dynamic RNNs for T-BPTT.
                # Splitting it up
                #print(data['sequence_lengths'])
                #total_timesteps = input_data.get_shape().as_list()[1]
                #num_slices = int(total_timesteps / self.backprop_length)
                #chunk_sizes = [self.backprop_length for i in range(num_slices)]
                #odd_size = total_timesteps - self.backprop_length*num_slices

                ## get all the chunks
                #if odd_size > 0:
                #    if b_stub_front:
                #        chunk_sizes = [odd_size] + chunk_sizes
                #    else:
                #        chunk_sizes = chunk_sizes + [odd_size]

                # Inplement TF style Truncated-backprop through time
                #self.train_step(model, input_data, output_data, active_entries, weights, chunk_sizes)
                print("begin training.")
                batch_loss = self.distributed_train_step(data) # , chunk_sizes
                #total_loss += self.distributed_train_step(input_data, output_data, active_entries, weights) # , chunk_sizes
                print("finish training.")
                total_loss += batch_loss
                print("finish sum.")
                num_batches += 1
          
            train_loss = total_loss / num_batches

            #for data in self.ds_valid:
            #    #self.valid_step(model, data['inputs'], data['outputs'], data['active_entries'])
            #    print("begin validing.")
            #    self.distributed_valid_step(data['inputs'], data['outputs'], data['active_entries'])
            #    print("finish validing.")

            #if tf.math.is_nan(self.valid_loss.result()):
            #    logging.warning("NAN Loss found, terminating routine")
            #    break

            # save history
            history['train_loss'].append(train_loss) # train_loss
            history['train_mse'].append(self.train_metric.result().numpy())
            history['valid_loss'].append(self.valid_loss.result().numpy())
            history['valid_mse'].append(self.valid_metric.result().numpy())


            # save optimal results
            if self.valid_loss.result() < min_loss and epoch > min_epochs:
                min_loss = self.valid_loss.result()
                save_model(self.model, params, history, option='optimal')

            # looging and state reset
            logs = 'Epoch={}/{},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{} | {}'
        
            if epoch%1 ==0:
                printbar()
                tf.print(tf.strings.format(logs,
                (epoch, self.epochs, train_loss, self.train_metric.result(),self.valid_loss.result(),self.valid_metric.result(), params['net_name']))) # train_loss
                tf.print("")

            #if epoch%2 == 0:
            #    checkpoint.save(os.path.join(self.checkpoint_dir, "ckpt"))
            
            #self.train_loss.reset_states()
            self.valid_loss.reset_states()
            self.train_metric.reset_states()
            self.valid_metric.reset_states()

        return history