"""Training remains gated until a separately authorized E2/E3 implementation review."""
def require_training(authorization=None):
    raise PermissionError('Real optimizer updates are not authorized by E1 or interface confirmation; E2/E3 require separate authorization')


if __name__=='__main__':
    require_training()
