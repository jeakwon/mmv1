import numpy as np

def correlation(pred, label, chunk_len=100, flatten_neuron=False):
    
    '''
    pred, label: arrays of shape (num_timesteps, num_neurons)
    
    return an array of correlation coefficients
    
    if flatten_neuron = True, the returned array is of shape 
    (num_timesteps//chunk_len), where each entry is the correlation
    coefficient between corresponding pred and label narrays of shape 
    (chunk_len * num_neurons)
    
    if flatten_neuron = False, the returned array is of shape 
    (num_timesteps//chunk_len, num_neurons), where each entry is the 
    correlation between corresponding pred and label arrays of shape
    (chunk_len)
    '''
    
    res = []
    for i in range(0, pred.shape[0], chunk_len):
        if flatten_neuron:
            pred_current_chunk = pred[i:i+chunk_len].flatten()
            label_current_chunk = label[i:i+chunk_len].flatten()
            cor_coef = np.corrcoef(pred_current_chunk, label_current_chunk)[0][1]
            res.append(cor_coef)
        else:
            res_per_neuron = []
            for j in range(pred.shape[1]):
                cor_coef = np.corrcoef(pred[i:i+chunk_len, j], label[i:i+chunk_len, j])[0][1]
                res_per_neuron.append(cor_coef)
            res.append(res_per_neuron)
    return np.array(res)

# correlation in time, averaged with neurons
def cor_in_time(pred, label):
    return correlation(pred, label, chunk_len=pred.shape[0], flatten_neuron=False)

# correlation in neurons, averaged with trials
def cor_in_neurons(pred, label):
    return correlation(pred, label, chunk_len=1, flatten_neuron=True)

def evaluate_model(model, weights_path, dataset, device):

    dl = DataLoader(dataset=dataset, batch_size=256, shuffle=False, num_workers=4)

    model.load_state_dict(torch.load(weights_path))

    ground_truth_all = []
    pred_all = []

    model.eval()

    with torch.no_grad():

        for (image, behav, spikes) in dl:

            image = image.to(device)
            behav = behav.to(device)

            pred = model(image, behav)

            ground_truth_all.append(spikes.numpy())
            pred_all.append(pred.cpu().numpy())

    return np.concatenate(pred_all, axis=0), np.concatenate(ground_truth_all, axis=0)