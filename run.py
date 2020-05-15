from tensorflow.keras.losses import categorical_crossentropy

from evaluation.adience import evaluate_adience_vgg, evaluate_adience_alxs


def main():
    evaluate_adience_vgg(categorical_crossentropy)
    evaluate_adience_alxs(categorical_crossentropy)


if __name__ == "__main__":
    main()
