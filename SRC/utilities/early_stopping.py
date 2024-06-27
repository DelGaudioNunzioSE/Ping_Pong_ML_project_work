
class EarlyStopping:
    def __init__(self, patience=20, restore_best_weights=True):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait after the last improvement in validation loss.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
        """
        self.patience = patience  # Number of epochs to wait for improvement
        self.restore_best_weights = restore_best_weights  # Flag to restore best weights
        self.best_val_loss = float('inf')  # Best validation loss observed
        self.epochs_no_improve = 0  # Counter for epochs with no improvement
        self.best_model_wts = None  # Best model weights observed
        self.early_stop = False  # Flag to indicate whether to stop early

    def __call__(self, current_val_loss, model):
        """
        Check if early stopping criteria are met.

        Args:
            current_val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model being trained.
        """
        if current_val_loss < self.best_val_loss:
            # Update the best validation loss and reset the counter
            self.best_val_loss = current_val_loss
            self.epochs_no_improve = 0
            if self.restore_best_weights:
                # Save the model weights
                self.best_model_wts = model.state_dict()
        else:
            # Increment the counter if no improvement
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            # Trigger early stopping if no improvement for 'patience' epochs
            self.early_stop = True
            if self.restore_best_weights and self.best_model_wts is not None:
                # Restore model weights if required
                model.load_state_dict(self.best_model_wts)
