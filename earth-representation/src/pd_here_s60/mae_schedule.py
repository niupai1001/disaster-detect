"""Learning-rate and validation timing for the MAE pretraining recipe."""
import math

WARMUP_EPOCHS = 8
BASE_LR = 1.5e-4
WEIGHT_DECAY = .05
BETAS = (.9, .95)


def mae_lr(peak, progress, epochs, warmup=WARMUP_EPOCHS):
    if progress < warmup:
        return peak * progress / warmup
    phase = (progress - warmup) / (epochs - warmup)
    return peak * (1 + math.cos(math.pi * phase)) / 2


def after_warmup(state, epoch, warmup=WARMUP_EPOCHS):
    # Keep all checkpoint scores. Only the patience clock starts at warmup end.
    if epoch <= warmup:
        state.update(reference=state['best_score'], reference_epoch=epoch,
                     stale_epochs=0, triggered=False)
    return state
