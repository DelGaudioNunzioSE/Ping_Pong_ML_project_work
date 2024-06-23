
class EarlyStopping:
    def __init__(self, patience=20, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_wts = None
        self.early_stop = False

    def __call__(self, current_val_loss, model):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.epochs_no_improve = 0
            if self.restore_best_weights:
                self.best_model_wts = model.state_dict()
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_model_wts is not None:
                model.load_state_dict(self.best_model_wts)
