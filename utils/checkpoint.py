import os
import torch


def save_checkpoint(model, optimizer, epoch, run_id, records, best_nmi, scheduler=None, filename='checkpoint.pth.tar'):
    """
    Save the model checkpoint along with the records.

    Parameters:
    - model: The model to save.
    - optimizer: The optimizer state to save.
    - epoch (int): The current epoch number.
    - run_id: The run ID for the experiment.
    - records: The records object to save.
    - best_nmi: The best NMI score so far.
    - scheduler: The learning rate scheduler state to save. Defaults to None.
    - filename (str): The path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'run_id': run_id,
        'records': records.records,
        'best_metrics': records.best_metrics,
        'best_nmi': best_nmi
    }
    
    if scheduler:
        state['scheduler'] = scheduler.state_dict()
    
    torch.save(state, filename)


def load_checkpoint(model, optimizer, records, filename='checkpoint.pth.tar', scheduler=None):
    """
    Load the model checkpoint along with the records.

    Parameters:
    - model: The model to load the state into.
    - optimizer: The optimizer to load the state into.
    - records: The records object to load the state into.
    - filename (str): The path to the checkpoint file.
    - scheduler: The learning rate scheduler to load the state into. Defaults to None.

    Returns:
    - start_epoch (int): The epoch to resume training from.
    - run_id: The run ID for the experiment.
    - best_nmi: The best NMI score so far.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file '{filename}' not found.")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    records.records = checkpoint.get('records', records.records)
    records.best_metrics = checkpoint.get('best_metrics', records.best_metrics)
    best_nmi = checkpoint.get('best_nmi', 0.0)
    start_epoch = checkpoint['epoch'] + 1
    run_id = checkpoint.get('run_id', None)
    
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return start_epoch, run_id, best_nmi
