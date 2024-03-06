class EarlyStopping:
    def __init__(self, patience: int):
        """Early stopping callback
        for preventing overfitting

        Parameters
        ----------
        patience : int
            Number of epochs to wait for loss
            improvement
        """
        self.patience = patience
        self.last_loss = 1  # current loss
        self.epochs = 0  # counter


    def step(self, loss: float) -> bool:
        """Update state of callback.

        Parameters
        ----------
        loss : float
            Current loss

        Returns
        -------
        bool
            Activate saving
        """
        if loss > self.last_loss:
            self.epochs += 1  # increase counter
        else:
            self.epochs = 0  # reset counter
        self.last_loss = loss
        if self.epochs == self.patience:
            return True  # activate callback
        else:
            return False
        

class ModelCheckpoint:
    def __init__(self, save_period: int):
        """Callback for periodical model saving

        Parameters
        ----------
        save_period : int
            Period for model saving
        """
        self.save_period = save_period
        self.epoch = 0  # counter


    def step(self) -> bool:
        """Update state of callback

        Returns
        -------
        bool
            Activate saving
        """
        self.epoch += 1  # increase counter
        if self.epoch == self.save_period:
            self.epoch = 0  # reset counter
            return True  # activate callback
        else:
            return False
