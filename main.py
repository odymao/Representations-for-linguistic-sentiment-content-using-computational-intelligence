# run and get the generated files in outputs dir
import data.main
import pos_tagging.main
import features.main
import coreNLP.main

def main():
    print("Creating data, labels and embedding matrix files")
    data.main.main(w2v_limited=False, pad=False)

    print("Creating POS files")
    pos_tagging.main.main()

    print("Creating features files")
    features.main.main()

    print("Using Standford's coreNLP to get sentiment distribution")
    coreNLP.main.main(pad_length=4)

    print("ok")

if __name__ == "__main__":
    main()
