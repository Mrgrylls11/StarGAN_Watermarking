# --- START OF FILE logger.py ---

# import tensorflow as tf # Keep tensorflow for TF 2.x
import tensorboardX # Or use from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        # Use tensorboardX or PyTorch's SummaryWriter
        self.writer = tensorboardX.SummaryWriter(log_dir)
        # OR: self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        # Use the writer's add_scalar method
        self.writer.add_scalar(tag, value, step)
        # Flushing is usually handled automatically or on close

    # You might want to add a close method if using PyTorch's SummaryWriter
    def close(self):
         if hasattr(self.writer, 'close'):
              self.writer.close()


# --- END OF FILE logger.py ---