import os
import torch

def save_checkpoint(model, optimizer, epoch, run_id, records, best_nmi, filename='checkpoint.pth.tar'):
    """
    Save the model checkpoint along with the records.

    Parameters:
    - model: The model to save.
    - optimizer: The optimizer state to save.
    - epoch (int): The current epoch number.
    - run_id: The run ID for the experiment.
    - records: The records object to save.
    - best_nmi: The best NMI score so far.
    - filename (str): The path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'run_id': run_id,
        'records': records.records,  # Save the records state
        'best_metrics': records.best_metrics,  # Save the best metrics
        'best_nmi': best_nmi  # Save the best NMI score
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, records, filename='checkpoint.pth.tar'):
    """
    Load the model checkpoint along with the records.

    Parameters:
    - model: The model to load the state into.
    - optimizer: The optimizer to load the state into.
    - records: The records object to load the state into.
    - filename (str): The path to the checkpoint file.

    Returns:
    - start_epoch (int): The epoch to resume training from.
    - run_id: The run ID for the experiment.
    - best_nmi: The best NMI score so far.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    records.records = checkpoint.get('records', records.records)  # Load the records state
    records.best_metrics = checkpoint.get('best_metrics', records.best_metrics)  # Load the best metrics
    best_nmi = checkpoint.get('best_nmi', 0.0)  # Load the best NMI score
    start_epoch = checkpoint['epoch'] + 1
    run_id = checkpoint.get('run_id', None)
    return start_epoch, run_id, best_nmi
