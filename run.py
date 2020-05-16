from tensorflow.keras.losses import categorical_crossentropy

from evaluation.adience import evaluate_adience_vgg_f, evaluate_adience_alxs, evaluate_adience_res


def main():
    evaluate_adience_vgg_f(categorical_crossentropy)
    evaluate_adience_alxs(categorical_crossentropy)
    evaluate_adience_res(categorical_crossentropy)


if __name__ == "__main__":
    main()
