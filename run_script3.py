from run import AdienceEmd

import tensorflow.keras.backend as k


if __name__ == "__main__":
    adience_emd = AdienceEmd()
    for j in range(7):
        k.clear_session()
        adience_emd.alxs(j, 4)
