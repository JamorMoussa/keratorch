from keras.src.optimizers import Adam as KerasAdam

__all__ = ["Adam",]

class Adam(KerasAdam):

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        super().__init__(
            learning_rate=lr, 
            beta_1=betas[0],
            beta_2=betas[1],
            epsilon=eps, 
            weight_decay=weight_decay,amsgrad=amsgrad,
            name="adam"
        )