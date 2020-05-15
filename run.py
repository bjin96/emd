from tensorflow.keras.losses import categorical_crossentropy

from evaluation.adience import evaluate_adience_vgg_f, evaluate_adience_alxs


def main():
    evaluate_adience_vgg_f(categorical_crossentropy)
    evaluate_adience_alxs(categorical_crossentropy)


if __name__ == "__main__":
    main()
